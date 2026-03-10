# Replace the existing cleanup() implementation in sidekick.py with the following.

import asyncio
from typing import Optional

class Sidekick:
    # ... existing __init__ and other methods ...

    async def _async_cleanup(self, timeout: Optional[float] = 10.0):
        """
        Async helper that cleanly closes Playwright resources.

        - Awaits browser.close() and playwright.stop() sequentially.
        - Uses a try/except to continue closing even if one call fails.
        - The optional timeout (seconds) can be used by callers that await this coroutine.
        """
        # If no resources, nothing to do
        if not self.browser and not self.playwright:
            return

        # Attempt to close browser first, then playwright.
        # We swallow individual errors to make the cleanup best-effort.
        try:
            if self.browser:
                # browser.close() is a coroutine
                await self.browser.close()
        except Exception as e:
            # Log or print minimal info; don't re-raise during cleanup
            print(f"[Sidekick.cleanup] error closing browser: {e}")

        try:
            if self.playwright:
                # playwright.stop() is a coroutine
                await self.playwright.stop()
        except Exception as e:
            print(f"[Sidekick.cleanup] error stopping playwright: {e}")

        # Clear references so repeated cleanup is a no-op
        self.browser = None
        self.playwright = None

    def cleanup(self):
        """
        Public cleanup method that safely schedules or runs _async_cleanup.

        Behavior:
        - If an asyncio event loop is running: schedule _async_cleanup as a background task
          and (optionally) wait a short amount of time for it to finish. This avoids
          calling asyncio.run() inside a running loop.
        - If no event loop is running: run the coroutine synchronously via asyncio.run().

        Key points / rationale:
        - We use a single coroutine to close both resources (avoid multiple asyncio.run()).
        - We schedule using loop.create_task() when inside an existing event loop; this
          avoids RuntimeError: asyncio.run() cannot be called from a running event loop.
        - We wait briefly for the task to complete so the process teardown is less likely
          to kill the child Playwright process before it finishes closing.
        """
        # Nothing to do if resources already cleared
        if not getattr(self, "browser", None) and not getattr(self, "playwright", None):
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop (e.g. Gradio). Schedule background task.
            task = loop.create_task(self._async_cleanup())

            # Optionally wait a short time for the cleanup to finish to reduce orphaning risk.
            # We do this non-blockingly by adding a done-callback and also calling
            # asyncio.create_task to wait if possible. If you prefer not to wait at all,
            # remove the waiting block.
            def _on_done(t):
                try:
                    exc = t.exception()
                    if exc:
                        print(f"[Sidekick.cleanup] cleanup task raised: {exc}")
                except asyncio.CancelledError:
                    print("[Sidekick.cleanup] cleanup task cancelled")

            task.add_done_callback(_on_done)

            # Try to give the task a short time to complete. This uses call_soon_threadsafe
            # to schedule a coroutine that awaits the task with a timeout. If the loop is fully
            # occupied or the process exits immediately, this will not block teardown.
            try:
                # Schedule a watcher coroutine that waits for the cleanup with timeout
                async def _watch_task(t, timeout=5.0):
                    try:
                        await asyncio.wait_for(t, timeout=timeout)
                    except asyncio.TimeoutError:
                        # Not fatal — just warn
                        print("[Sidekick.cleanup] cleanup did not complete within timeout")
                    except Exception as e:
                        print(f"[Sidekick.cleanup] cleanup watcher caught: {e}")

                # create_task ensures the watcher runs in the same loop
                loop.create_task(_watch_task(task, timeout=5.0))
            except Exception as e:
                # Worst-case: scheduling the watcher failed — continue, the main task still runs.
                print(f"[Sidekick.cleanup] failed to schedule cleanup watcher: {e}")
        else:
            # No running loop: run the async cleanup synchronously.
            # This creates a fresh event loop, runs cleanup to completion, then closes it.
            try:
                asyncio.run(self._async_cleanup())
            except Exception as e:
                print(f"[Sidekick.cleanup] error running cleanup synchronously: {e}")

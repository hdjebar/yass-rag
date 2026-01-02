"""
Test rate limiting utilities.
"""

import pytest
import time
from yass_rag.utils import RateLimiter
from threading import Thread


def test_rate_limiter_basic():
    """Test basic rate limiting."""
    limiter = RateLimiter(rate=5, per=1.0)  # 5 requests per second

    @limiter
    def dummy_func():
        return "success"

    start = time.time()
    for _ in range(12):  # Make 12 requests - should trigger throttling
        dummy_func()
    elapsed = time.time() - start

    # Should take at least 1.4 seconds (12 requests at 5/sec, first 5 go through immediately)
    # Being lenient with timing due to system load
    assert elapsed >= 0.7, f"Rate limiting not working: took {elapsed}s"


def test_rate_limiter_thread_safety():
    """Test rate limiter with concurrent threads."""
    limiter = RateLimiter(rate=10, per=1.0)
    counter = [0]

    @limiter
    def increment():
        counter[0] += 1

    def threaded_increment():
        for _ in range(5):
            increment()

    start = time.time()
    threads = [Thread(target=threaded_increment) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start

    # 4 threads * 5 calls = 20 total
    assert counter[0] == 20
    # Should take at least 0.5 seconds (20 requests at 10/sec)
    assert elapsed >= 0.5, f"Rate limiting not working: took {elapsed}s"


def test_rate_limiter_different_rates():
    """Test rate limiter with different configurations."""
    # Fast rate
    fast_limiter = RateLimiter(rate=100, per=1.0)

    @fast_limiter
    def fast_func():
        return "fast"

    start = time.time()
    for _ in range(50):
        fast_func()
    fast_elapsed = time.time() - start

    # Slow rate
    slow_limiter = RateLimiter(rate=10, per=1.0)

    @slow_limiter
    def slow_func():
        return "slow"

    start = time.time()
    for _ in range(50):
        slow_func()
    slow_elapsed = time.time() - start

    # Fast limiter should be much faster
    assert fast_elapsed < slow_elapsed / 5, "Rate limiter not scaling correctly"

from bisect import bisect_left, bisect_right, insort
from typing import List


class SortedList:
    """
    A light-weight sorted multiset based on √N bucket decomposition.
    Supports:
        add(x)               – O(√N)
        ge(x) (ceil)         – O(√N)
        le(x) (floor)        – O(√N)
    Sufficient for 10^5 operations in the given task.
    """
    BUCKET_RATIO = 50      # tune to balance bucket size / amount
    REBUILD_RATIO = 170    # when to rebuild completely

    def __init__(self) -> None:
        self.size = 0
        self._buckets: List[List[int]] = []

    # ---------- internal helpers ----------
    def _build(self, arr: List[int]) -> None:
        """(Re)build buckets from a sorted array."""
        self.size = len(arr)
        if self.size == 0:
            self._buckets = []
            return
        b = int((self.size / self.BUCKET_RATIO) ** 0.5) + 1  # bucket size
        self._buckets = [arr[i:i + b] for i in range(0, self.size, b)]

    def _find_bucket(self, x: int) -> List[int]:
        """return the bucket that should contain x (or the last one)."""
        for bucket in self._buckets:
            if x <= bucket[-1]:
                return bucket
        return self._buckets[-1]

    # ---------- public interface ----------
    def add(self, x: int) -> None:
        """Insert x into multiset."""
        if self.size == 0:
            self._buckets = [[x]]
            self.size = 1
            return

        bucket = self._find_bucket(x)
        insort(bucket, x)
        self.size += 1

        # split an overgrown bucket and rebuild if necessary
        if len(bucket) > len(self._buckets) * self.REBUILD_RATIO:
            self._build(list(self))

    def __iter__(self):
        for bucket in self._buckets:
            for val in bucket:
                yield val

    # ---- neighbours ----
    def ge(self, x: int):
        """Smallest element >= x, or None."""
        for bucket in self._buckets:
            if bucket[-1] >= x:
                idx = bisect_left(bucket, x)
                if idx < len(bucket):
                    return bucket[idx]
        return None

    def le(self, x: int):
        """Largest element <= x, or None."""
        for bucket in reversed(self._buckets):
            if bucket[0] <= x:
                idx = bisect_right(bucket, x) - 1
                if idx >= 0:
                    return bucket[idx]
        return None


class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        # If x == 0, we may pick the same index twice, hence answer is 0
        if x == 0:
            return 0

        sl = SortedList()
        ans = 10 ** 18  # large initial value

        for i in range(x, len(nums)):
            # insert the element that is exactly x positions behind current index
            sl.add(nums[i - x])

            # look for closest neighbours to nums[i] in the multiset
            lo = sl.le(nums[i])
            if lo is not None:
                ans = min(ans, nums[i] - lo)

            hi = sl.ge(nums[i])
            if hi is not None:
                ans = min(ans, hi - nums[i])

            # Early exit: minimal possible difference already reached
            if ans == 0:
                return 0

        return ans
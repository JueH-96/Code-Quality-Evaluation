from collections import defaultdict
from heapq import nlargest

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if k >= n:
            return 0
        mid = n // 2
        left = defaultdict(int)
        right = defaultdict(int)
        for i in range(mid):
            left[nums[i]] += 1
            right[nums[n - i - 1]] += 1
        left = nlargest(k, left.items(), key=lambda x: x[1])
        right = nlargest(k, right.items(), key=lambda x: x[1])
        left = [v for _, v in left]
        right = [v for _, v in right]
        ans = n - left[0] - right[0] + (nums[0] + nums[-1]) > k
        for i, j in zip(left, right):
            if i + j <= k:
                break
            ans += 1
        if (n % 2) and nums[mid] > k:
            ans += 1
        return ans
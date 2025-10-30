class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] | nums[i]
        ans = float('inf')
        deque = []
        for i, val in enumerate(prefix):
            while deque and prefix[deque[-1]] >= val:
                deque.pop()
            while deque and prefix[i] - prefix[deque[0]] >= k:
                ans = min(ans, i - deque[0])
                deque.pop(0)
            deque.append(i)
        return ans if ans != float('inf') else -1
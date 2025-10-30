class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_len = float('inf')
        prefix_or = [0] * (n + 1)

        for i in range(n):
            prefix_or[i + 1] = prefix_or[i] | nums[i]

        for i in range(n):
            if prefix_or[i] >= k:
                min_len = min(min_len, i)

        for i in range(n):
            for j in range(i + min_len, n + 1):
                if prefix_or[j] - prefix_or[i] >= k:
                    min_len = min(min_len, j - i)

        return min_len if min_len != float('inf') else -1
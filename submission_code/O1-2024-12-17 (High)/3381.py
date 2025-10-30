class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        INF = float('inf')
        answer = INF
        
        for start in range(n):
            or_value = 0
            for end in range(start, n):
                or_value |= nums[end]
                if or_value >= k:
                    answer = min(answer, end - start + 1)
                    break
        
        return answer if answer != INF else -1
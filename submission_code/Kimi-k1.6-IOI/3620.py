class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        prev = -float('inf')
        count = 0
        for num in nums:
            current_low = num - k
            current_high = num + k
            next_possible = max(current_low, prev + 1)
            if next_possible <= current_high:
                count += 1
                prev = next_possible
        return count
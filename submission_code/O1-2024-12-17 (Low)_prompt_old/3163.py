class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        
        for i in range(n):
            current_set = set()
            for j in range(i, n):
                current_set.add(nums[j])
                distinct_count = len(current_set)
                total_sum += distinct_count * distinct_count
                
        return total_sum
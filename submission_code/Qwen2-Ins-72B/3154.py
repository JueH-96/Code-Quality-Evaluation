class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        max_diff = 0
        max_val = 0
        max_triplet = 0
        
        for num in nums:
            max_triplet = max(max_triplet, max_diff * num)
            max_diff = max(max_diff, max_val - num)
            max_val = max(max_val, num)
        
        return max_triplet
class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        # pos = max cost where the last element contributes positively
        # neg = max cost where the last element contributes negatively
        pos = 0
        neg = float('-inf')
        
        for num in nums:
            new_pos = max(pos, neg) + num
            new_neg = pos - num
            pos, neg = new_pos, new_neg
        
        return max(pos, neg)
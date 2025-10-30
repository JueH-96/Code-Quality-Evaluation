class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        from itertools import combinations
        
        n = len(nums)
        max_val = 0
        
        # Choose 2*k elements from nums
        for subsequence in combinations(range(n), 2 * k):
            subseq_nums = [nums[i] for i in subsequence]
            
            # Try all ways to split these 2*k elements into two groups of k each
            for group1_indices in combinations(range(2 * k), k):
                group1_or = 0
                group2_or = 0
                
                group1_set = set(group1_indices)
                
                for i in range(2 * k):
                    if i in group1_set:
                        group1_or |= subseq_nums[i]
                    else:
                        group2_or |= subseq_nums[i]
                
                max_val = max(max_val, group1_or ^ group2_or)
        
        return max_val
class Solution:
    def canSortArray(self, nums):
        n = len(nums)
        c_i = []  # set bits counts at positions
        from collections import defaultdict

        # compute set bits counts
        def count_set_bits(x):
            return bin(x).count('1')
        
        c_i = [count_set_bits(nums[i]) for i in range(n)]
        
        # group nums[i] by their set bits counts
        group_nums = defaultdict(list)
        group_positions = defaultdict(list)
        
        for i in range(n):
            k = c_i[i]
            group_nums[k].append(nums[i])
            group_positions[k].append(i)
        
        # sort the nums in each group
        for k in group_nums:
            group_nums[k].sort()
        
        # assign the nums[i] to positions in their groups
        assembled = [0]*n
        group_indices = {k:0 for k in group_nums}  # indices for each group's nums
        
        for i in range(n):
            k = c_i[i]
            idx = group_indices[k]
            assembled[i] = group_nums[k][idx]
            group_indices[k] +=1  # move to next element in group
        
        # check if assembled array is sorted
        for i in range(1, n):
            if assembled[i] < assembled[i-1]:
                return False
        return True
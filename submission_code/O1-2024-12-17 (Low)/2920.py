class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        # If all elements are already equal, no operations needed
        if all(x == nums[0] for x in nums):
            return 0
        
        from collections import defaultdict
        
        # positions[val] will store all indices where nums[index] == val
        positions = defaultdict(list)
        for i, val in enumerate(nums):
            positions[val].append(i)
        
        # We will compute, for each distinct value, how many seconds are needed
        # for that value to "spread" to the entire array. We'll then take
        # the minimum of such times over all values.
        # 
        # For a chosen value v, let p0 < p1 < ... < p(k-1) be the sorted
        # positions where v appears. We consider the circular gaps:
        #    dist_i = p(i+1) - p(i) for i in [0..k-2]
        #    and dist_{wrap} = (p0 + n) - p(k-1)
        # 
        # The time needed is max(dist_i)//2 over these gaps.
        # We'll pick the minimal time over all possible values v.
        
        ans = float('inf')
        
        for val, idx_list in positions.items():
            idx_list.sort()
            max_gap = 0
            k = len(idx_list)
            
            for i in range(k - 1):
                # distance between consecutive occurrences
                gap = idx_list[i+1] - idx_list[i]
                max_gap = max(max_gap, gap)
            
            # wrap-around distance
            wrap_gap = (idx_list[0] + n) - idx_list[-1]
            max_gap = max(max_gap, wrap_gap)
            
            # Time needed for this value is max_gap//2
            ans = min(ans, max_gap // 2)
        
        return ans
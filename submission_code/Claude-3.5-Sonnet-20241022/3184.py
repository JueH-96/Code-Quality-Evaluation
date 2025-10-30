class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        # Transform nums[i] - i to find increasing subsequences
        transformed = [(nums[i] - i, i) for i in range(n)]
        transformed.sort()
        
        # Binary indexed tree to maintain maximum sums
        bit = {}
        
        def update(idx, val):
            while idx < n:
                bit[idx] = max(bit.get(idx, float('-inf')), val)
                idx = idx | (idx + 1)
                
        def query(idx):
            res = float('-inf')
            while idx >= 0:
                res = max(res, bit.get(idx, float('-inf')))
                idx = (idx & (idx + 1)) - 1
            return res
        
        ans = float('-inf')
        
        # Process each element in sorted order of nums[i] - i
        for diff, i in transformed:
            # Get maximum sum possible before index i
            curr = query(i)
            # New sum will be nums[i] + max sum before i
            new_sum = nums[i] if curr == float('-inf') else nums[i] + curr
            # Update answer and BIT
            ans = max(ans, new_sum)
            update(i, new_sum)
            
        return ans
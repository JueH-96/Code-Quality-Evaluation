class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        # We will try removing i elements from the front (i in [0..n]),
        # and if the remaining subarray is distinct, compute the cost of
        # removing those i elements in steps of 3. We then take the minimum
        # among all valid i.
        
        # Helper function to check if a list has all distinct elements
        def is_distinct(subarr: List[int]) -> bool:
            return len(subarr) == len(set(subarr))
        
        min_ops = float('inf')
        for i in range(n+1):
            # Subarray from i to end
            sub = nums[i:]
            if is_distinct(sub):
                # Cost is the number of operations needed to remove i elements
                # Each operation can remove up to 3 elements
                cost = (i + 2) // 3  # Equivalent to math.ceil(i / 3)
                min_ops = min(min_ops, cost)
        
        return min_ops
class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Create a list of tuples (nums[i] - i, i)
        arr = [(nums[i] - i, i) for i in range(n)]
        
        # Sort the list based on nums[i] - i
        arr.sort()
        
        # Binary Indexed Tree (Fenwick Tree) for efficient range sum queries
        bit = [0] * (n + 1)
        
        def update(index, val):
            index += 1
            while index <= n:
                bit[index] = max(bit[index], val)
                index += index & (-index)
        
        def query(index):
            index += 1
            result = 0
            while index > 0:
                result = max(result, bit[index])
                index -= index & (-index)
            return result
        
        # Mapping of original indices to sorted indices
        index_map = {v[1]: i for i, v in enumerate(arr)}
        
        max_sum = float('-inf')
        
        for i in range(n):
            orig_index = arr[i][1]
            current_sum = nums[orig_index] + query(index_map[orig_index])
            max_sum = max(max_sum, current_sum)
            update(index_map[orig_index], current_sum)
        
        return max_sum
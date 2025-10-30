import bisect

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [-float('inf')] * (self.n + 1)  # Initialize with negative infinity

    def update(self, index, value):
        while index <= self.n:
            if value > self.tree[index]:
                self.tree[index] = value
            else:
                break  # No need to proceed if the current value isn't updated
            index += index & -index

    def query(self, index):
        res = -float('inf')
        while index > 0:
            if self.tree[index] > res:
                res = self.tree[index]
            index -= index & -index
        return res

class Solution:
    def maxBalancedSubsequenceSum(self, nums: list) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        keys = [nums[i] - i for i in range(n)]
        unique_sorted = sorted(set(keys))
        size = len(unique_sorted)
        
        ft = FenwickTree(size)
        
        max_sum = -float('inf')
        
        for i in range(n):
            key_i = keys[i]
            
            # Find the number of keys <= key_i
            idx = bisect.bisect_right(unique_sorted, key_i)
            
            # Query the maximum in the range [1, idx]
            current_max = ft.query(idx)
            
            if current_max == -float('inf'):
                dp_i = nums[i]
            else:
                option1 = current_max + nums[i]
                option2 = nums[i]
                dp_i = max(option1, option2)
            
            if dp_i > max_sum:
                max_sum = dp_i
            
            # Find the position in unique_sorted where key_i is located
            pos = bisect.bisect_left(unique_sorted, key_i)
            # The compressed index is pos + 1 (since Fenwick tree is 1-based)
            compressed_index = pos + 1
            ft.update(compressed_index, dp_i)
        
        return max_sum if max_sum != -float('inf') else 0
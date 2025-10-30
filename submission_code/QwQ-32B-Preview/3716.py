class FenwickTree:
    def __init__(self, size):
        self.N = size + 2
        self.tree = [0] * (self.N)
    
    def update(self, index, value):
        while index < self.N:
            if value > self.tree[index]:
                self.tree[index] = value
            index += index & -index
    
    def query_max(self, left, right):
        max_val = 0
        while right > left:
            if right & 1:
                val = self.tree[right - 1]
                if val > max_val:
                    max_val = val
                right -= 1
            right >>= 1
        while left < right:
            if left & 1:
                val = self.tree[left - 1]
                if val > max_val:
                    max_val = val
                left += 1
            left >>= 1
        return max_val

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return n
        
        # Group indices by number values
        prev_indices = {}
        for idx, num in enumerate(nums):
            if num in prev_indices:
                prev_indices[num].append(idx)
            else:
                prev_indices[num] = [idx]
        
        # Initialize Fenwick tree and dp table
        fenwick = FenwickTree(300)
        dp = [[0 for _ in range(300)] for _ in range(n)]
        max_length = 1
        
        for i in range(n):
            for d in range(300):
                prev_num1 = nums[i] + d
                prev_num2 = nums[i] - d
                max_len_from_j = 0
                
                # Check prev_num1
                if 1 <= prev_num1 <= 300:
                    if prev_num1 in prev_indices:
                        for j in prev_indices[prev_num1]:
                            if j < i:
                                max_len_from_j = max(max_len_from_j, fenwick.query_max(d, 299))
                
                # Check prev_num2
                if 1 <= prev_num2 <= 300:
                    if prev_num2 in prev_indices:
                        for j in prev_indices[prev_num2]:
                            if j < i:
                                max_len_from_j = max(max_len_from_j, fenwick.query_max(d, 299))
                
                if max_len_from_j != 0:
                    dp[i][d] = max_len_from_j + 1
                else:
                    dp[i][d] = 1
                
                # Update Fenwick tree
                fenwick.update(d, dp[i][d])
                
                # Update global maximum length
                if dp[i][d] > max_length:
                    max_length = dp[i][d]
        
        return max_length
from typing import List
import bisect

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        class SegmentTree:
            def __init__(self, size):
                self.N = 1
                while self.N < size:
                    self.N <<= 1
                self.size = self.N
                self.tree = [float('-inf')] * (2 * self.N)
            
            def update(self, idx, value):
                idx += self.N
                self.tree[idx] = max(self.tree[idx], value)
                while idx > 1:
                    idx >>= 1
                    self.tree[idx] = max(self.tree[2*idx], self.tree[2*idx + 1])
            
            def query(self, l, r):
                res = float('-inf')
                l += self.N
                r += self.N
                while l < r:
                    if l & 1:
                        res = max(res, self.tree[l])
                        l += 1
                    if r & 1:
                        r -= 1
                        res = max(res, self.tree[r])
                    l >>= 1
                    r >>= 1
                return res
        
        n = len(nums)
        keys = [num - idx for idx, num in enumerate(nums)]
        sorted_keys = sorted(list(set(keys)))
        key_to_idx = {key: idx for idx, key in enumerate(sorted_keys)}
        
        st = SegmentTree(len(sorted_keys))
        dp = [0] * n
        max_sum = float('-inf')
        
        for i in range(n):
            K = nums[i] - i
            # Find rightmost index where key[j] <= K
            idx = bisect.bisect_right(sorted_keys, K) - 1
            if idx >= 0:
                current_max = st.query(0, idx + 1)
                if current_max != float('-inf'):
                    dp[i] = nums[i] + current_max
                else:
                    dp[i] = nums[i]
            else:
                dp[i] = nums[i]
            # Update the segment tree with dp[i] at key[j] = nums[i] - i
            key = nums[i] - i
            st.update(key_to_idx[key], dp[i])
            max_sum = max(max_sum, dp[i])
        
        return max_sum
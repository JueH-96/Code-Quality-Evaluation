from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        q = len(queries)
        
        # Compute prefix_xor
        prefix_xor = [0] * (n + 1)
        for i in range(n):
            prefix_xor[i + 1] = prefix_xor[i] ^ nums[i]
        
        # Precompute max_f for each l
        max_f = [0] * n
        for l in range(n):
            current_max = 0
            for y in range(l, n):
                length = y - l + 1
                if length % 2 == 1:
                    # Odd length: f(l, y) = prefix_xor[y+1] ^ prefix_xor[l]
                    f = prefix_xor[y + 1] ^ prefix_xor[l]
                else:
                    # Even length: f(l, y) = prefix_xor[y] ^ prefix_xor[l]
                    f = prefix_xor[y] ^ prefix_xor[l]
                if f > current_max:
                    current_max = f
            max_f[l] = current_max
        
        # Build a segment tree for range maximum queries
        # Segment tree size is 2*(next power of two of n) - 1
        size = 1
        while size < n:
            size <<= 1
        seg = [0] * (2 * size)
        
        # Initialize leaf nodes
        for i in range(n):
            seg[size + i] = max_f[i]
        # Build the tree
        for i in range(size - 1, 0, -1):
            seg[i] = max(seg[2 * i], seg[2 * i + 1])
        
        # Function to query the maximum in range [l, r)
        def query(l, r):
            l += size
            r += size
            res = 0
            while l < r:
                if l % 2 == 1:
                    res = max(res, seg[l])
                    l += 1
                if r % 2 == 1:
                    r -= 1
                    res = max(res, seg[r])
                l //= 2
                r //= 2
            return res
        
        # Answer the queries
        answer = []
        for l, r in queries:
            # The maximum subarray XOR score within [l, r] is the max of max_f[l to r]
            ans = query(l, r + 1)
            answer.append(ans)
        
        return answer
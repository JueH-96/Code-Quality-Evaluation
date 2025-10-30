from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        # A very low number used to denote impossible (−∞ equivalent)
        NEG_INF = -10**15
        
        # merge two 2x2 DP matrices.
        # Each node (matrix) represents a transformation:
        # For input state i (i=0 for "free" and 1 for "forced skip"),
        # the node provides two outputs:
        #   dp[i][0]: maximum added profit finishing in state 0 (last not chosen)
        #   dp[i][1]: maximum added profit finishing in state 1 (last chosen)
        # For merge, if we have left and right segments, then:
        #   res[i][j] = max_{k in {0,1}} [ left[i][k] + right[k][j] ]
        def merge(left: List[List[int]], right: List[List[int]]) -> List[List[int]]:
            res = [[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]]
            for i in range(2):
                # Case: intermediate state k = 0
                cand = left[i][0] + right[0][0]
                if cand > res[i][0]:
                    res[i][0] = cand
                cand = left[i][0] + right[0][1]
                if cand > res[i][1]:
                    res[i][1] = cand
                # Case: intermediate state k = 1
                cand = left[i][1] + right[1][0]
                if cand > res[i][0]:
                    res[i][0] = cand
                cand = left[i][1] + right[1][1]
                if cand > res[i][1]:
                    res[i][1] = cand
            return res
        
        # Identity transformation for an empty segment.
        # An empty segment does nothing so the state remains the same,
        # and no profit is added.
        identity = [[0, NEG_INF], [NEG_INF, 0]]
        
        # We build an iterative segment tree over the array.
        # First, set base to be the smallest power-of-2 not less than n.
        base = 1 << ((n - 1).bit_length())
        size = 2 * base
        tree = [None] * size
        
        # For a single element a, if we are free (state 0), we have two choices:
        # • Skip it: profit = 0, remains state 0.
        # • Take it: profit = a, and state becomes 1.
        # And if we are forced to skip (state 1), our only option is to skip:
        # profit = 0, and state becomes 0.
        # So the transformation matrix (leaf) for value a is:
        #    [[0, a],
        #     [0, NEG_INF]]
        for i in range(base):
            if i < n:
                tree[base + i] = [[0, nums[i]], [0, NEG_INF]]
            else:
                tree[base + i] = identity
                
        # Build the internal nodes.
        for i in range(base - 1, 0, -1):
            tree[i] = merge(tree[2 * i], tree[2 * i + 1])
        
        ans = 0
        # Process each query:
        # For query [pos, x]:
        #   1. Update nums[pos] to x (i.e. update the leaf transformation).
        #   2. Rebuild the segment tree path up from that leaf.
        #   3. The answer for the full array is given by applying the root transformation
        #      to an initial free state (state 0). So the best profit is max( dp[0][0], dp[0][1] ).
        for pos, x in queries:
            # Update the leaf node for the changed element.
            tree[base + pos] = [[0, x], [0, NEG_INF]]
            idx = (base + pos) // 2
            while idx >= 1:
                tree[idx] = merge(tree[2 * idx], tree[2 * idx + 1])
                idx //= 2
            # The full-transformation from an initial free state (state 0) is stored at tree[1].
            curr = tree[1][0]
            current_ans = curr[0] if curr[0] >= curr[1] else curr[1]
            # As empty subsequence is allowed, the answer cannot be negative.
            if current_ans < 0:
                current_ans = 0
            ans = (ans + current_ans) % mod
        return ans

# --- Sample testing ---
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    nums = [3, 5, 9]
    queries = [[1, -2], [0, -3]]
    print(sol.maximumSumSubsequence(nums, queries))  # Expected output: 21

    # Example 2:
    nums = [0, -1]
    queries = [[0, -5]]
    print(sol.maximumSumSubsequence(nums, queries))  # Expected output: 0
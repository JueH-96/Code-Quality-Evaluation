from typing import List

MOD = 10**9 + 7
INF = -10**15   # a very negative number to represent -infinity

# In this solution we will build a segment tree that “merges” intervals under the rules
# of the house robber (non-adjacent selection) problem.
#
# We want to answer: given an array nums (which can be updated on single positions),
# what is the maximum sum of a subsequence with no two adjacent elements (an empty subsequence is allowed)
# This is the typical “house robber” problem.
#
# The idea is to represent each segment (an interval of the array) as a transformation from
# an "input" state to an "output" state along with the maximum sum that can be obtained.
#
# We define a state variable that indicates whether the previous element (just before the segment)
# was taken or not:
#   • 0 means the previous element was not taken ("free" to pick the first element in this segment).
#   • 1 means the previous element was taken (so we cannot choose the first element in the segment).
#
# For a segment covering some indices, we wish to store a 2x2 matrix dp such that:
#   dp[input_state][output_state] = maximum sum obtainable in that segment when:
#     • You start with input_state (0: free; 1: forced skip for first element)
#     • And after processing the whole segment, the “state” is output_state.
#
# Here the output state will be taken as follows:
#   • 0 means that the last element in the segment was not taken (thus the next element is free).
#   • 1 means that the last element was taken (thus the next element will be forced to skip).
#
# For a single element with value v, we construct the leaf (a segment of length one) as follows:
#
#   If the input state is free (0):
#     • You may choose to skip this element: contribution 0; output state remains free.
#     • Or if v is positive (choosing a negative value is never advantageous
#       since skipping is always an option yielding 0), you may take it:
#         contribution v; output state becomes 1 (because you took it).
#
#   If the input state is forced (1):
#     • You are not allowed to take the element, so you must skip:
#         contribution 0; output state becomes free.
#
# Thus, the leaf node matrix looks like:
#
#   leaf[v] = [ [0        , (v if v>0 else -INF) ],
#               [0        , -INF                  ] ]
#
# When merging two segments (with matrices A and B), the combined segment’s transformation is:
#
#   For every input state i and output state j:
#
#       merged[i][j] = max ( over k in {0,1} ) { A[i][k] + B[k][j] }
#
# In the segment tree we build leaves for each index, and update them as queries ask.
#
# After each update we compute the answer (maximum sum for the entire array)
# which is given by starting with a free state. That is: answer = max(root[0][0], root[0][1]).
#
# Finally, we aggregate all query answers and return (result mod 10^9 + 7).
#
class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        size = 1
        while size < n:
            size *= 2
        # Create a segment tree as an array "seg" of size 2*size.
        # Each node is a 2x2 matrix:
        #   dp[i][j] where i is the input state (0 or 1),
        #   and j is the final state (0: free, 1: forced skip because last element was taken).
        seg = [[[INF, INF], [INF, INF]] for _ in range(2 * size)]
        
        # Function to build a leaf matrix from a given value v.
        def make_leaf(v: int) -> List[List[int]]:
            # Initialize matrix with -INF (for impossible transitions)
            # For forced input state (1), only option is to skip:
            #    dp[1][0] = 0 and dp[1][1] remains impossible.
            ret = [[0, INF], [0, INF]]
            # For free input state (0): two choices
            # Option 1: Skip the element -> contribution 0, state remains free.
            # Option 2: Take the element, if v > 0 -> contribution v, state becomes forced skip.
            if v > 0:
                ret[0][1] = v
            else:
                ret[0][1] = INF
            return ret
        
        # Build leaves for positions 0 to n-1.
        for i in range(n):
            seg[size + i] = make_leaf(nums[i])
        # For indices beyond n (dummy leaves), we set them as an identity (empty segment)
        for i in range(n, size):
            # An empty segment “does nothing”: regardless of input state, output state free with sum 0.
            seg[size + i] = [[0, INF], [0, INF]]
        
        # Merge function: combine two matrices A and B.
        def merge(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
            res = [[INF, INF], [INF, INF]]
            for i in range(2):
                for j in range(2):
                    best = INF
                    for k in range(2):
                        candidate = A[i][k] + B[k][j]
                        if candidate > best:
                            best = candidate
                    res[i][j] = best
            return res
        
        # Build the segment tree from leaves upward.
        for i in range(size - 1, 0, -1):
            seg[i] = merge(seg[2 * i], seg[2 * i + 1])
        
        # Function to update the position pos (0-indexed) with new value newVal.
        def update(pos: int, newVal: int) -> None:
            idx = pos + size
            seg[idx] = make_leaf(newVal)
            idx //= 2
            while idx:
                seg[idx] = merge(seg[2 * idx], seg[2 * idx + 1])
                idx //= 2
        
        total_ans = 0
        # Process each query.
        for pos, x in queries:
            update(pos, x)
            root = seg[1]   # overall segment matrix
            # Answer is computed by starting with free state (0):
            curr = max(root[0][0], root[0][1])
            # The DP always allows the empty subsequence, so answer is at least 0.
            if curr < 0:
                curr = 0
            total_ans = (total_ans + curr) % MOD
        
        return total_ans

# For testing purposes:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.maximumSumSubsequence([3,5,9], [[1,-2],[0,-3]]))  # Expected output: 21
    # Example 2:
    print(sol.maximumSumSubsequence([0,-1], [[0,-5]]))  # Expected output: 0
from typing import List

INF_NEG = -10**18

# Each segment tree node stores a 2x2 matrix M so that:
# M[i][j] = maximum additional sum we can get from that segment
# if the segment is started with state i and finished with state j.
#
# Here our state is:
#   0: previous element was not selected
#   1: previous element WAS selected (so we cannot select the current element)
#
# For a leaf corresponding to a single element with value v:
#   if initial state is 0, we have two options:
#       - Skip: then you get sum 0 and the end state is 0.
#       - Take: allowed only if v is beneficial; note that if v is negative best option is to skip.
#         However if we take then the end state becomes 1 (because we took this element).
#         So the corresponding value is v. But if v is negative we never want to choose it.
#         So we record it as v, but note the combination later will choose maximum.
#   if initial state is 1, we are forced to skip, so the only possibility is (state becomes 0, sum = 0).
#
# Therefore for a leaf, we set:
#   M[0][0] = 0            (skip)
#   M[0][1] = v            (take)  [it might be negative, but then the max will choose skip]
#   M[1][0] = 0            (forced skip)
#   M[1][1] = INF_NEG      (invalid)
#
# When merging two segments with matrices A and B, we have:
#   C = A combined with B       (i.e., C[i][j] = max_{k in {0,1}} (A[i][k] + B[k][j]) )
# Finally, the answer for the current full nums array is:
#   answer = max( S[0][0], S[0][1] )
#
# When a point update is performed (nums[pos] becomes x), we update the corresponding leaf matrix.
    
class SegmentTree:
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        # the segment tree will be 1-indexed. size = 4*n
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.tree = [ [[INF_NEG, INF_NEG], [INF_NEG, INF_NEG]] for _ in range(2 * self.size) ]
        self.build(arr)
    
    def merge(self, A, B):
        # Merge two matrices A and B: result C of size 2x2.
        C = [[INF_NEG, INF_NEG],[INF_NEG, INF_NEG]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    C[i][j] = max(C[i][j], A[i][k] + B[k][j])
        return C
        
    def build(self, arr: List[int]):
        # Build leaves
        for i in range(self.n):
            v = arr[i]
            idx = i + self.size
            # For initial state 0, two options: skipping yields sum 0 with end state 0; taking yields v with end state 1.
            self.tree[idx][0][0] = 0
            self.tree[idx][0][1] = v
            # For initial state 1, we are forced to skip.
            self.tree[idx][1][0] = 0
            self.tree[idx][1][1] = INF_NEG
        # For positions beyond n, set identity-like transformation.
        for i in range(self.n, self.size):
            idx = i + self.size
            # Identity: doing nothing: from any state i -> same state with addition 0.
            self.tree[idx][0][0] = 0
            self.tree[idx][0][1] = INF_NEG
            self.tree[idx][1][0] = 0
            self.tree[idx][1][1] = INF_NEG
            
        for i in range(self.size-1, 0, -1):
            self.tree[i] = self.merge(self.tree[2*i], self.tree[2*i+1])
    
    def update(self, pos: int, value: int):
        idx = pos + self.size
        # update the leaf matrix for new value.
        self.tree[idx][0][0] = 0
        self.tree[idx][0][1] = value
        self.tree[idx][1][0] = 0
        self.tree[idx][1][1] = INF_NEG
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.merge(self.tree[2*idx], self.tree[2*idx+1])
            idx //= 2
            
    def query_total(self):
        # our answer is max(tree[1][0][0], tree[1][0][1]) since initial state is 0.
        root = self.tree[1]
        return max(root[0][0], root[0][1])
    
class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        mod = 10**9+7
        seg = SegmentTree(nums)
        total = 0
        for pos, x in queries:
            seg.update(pos, x)
            res = seg.query_total()
            total = (total + res) % mod
        return total

# ----------------------
# For local testing:

if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    nums = [3,5,9]
    queries = [[1,-2],[0,-3]]
    print(sol.maximumSumSubsequence(nums, queries))  # Expected output: 21
    
    # Example 2:
    nums = [0,-1]
    queries = [[0,-5]]
    print(sol.maximumSumSubsequence(nums, queries))  # Expected output: 0
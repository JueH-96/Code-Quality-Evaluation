from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        """
        We maintain a segment tree where each node stores information about
        the maximum sum of a non-adjacent subsequence in its range, together
        with boundary "pick" states:
        
          st(a1,a2) where:
            a1 = 0 or 1 indicating whether the first element in this segment is NOT picked or IS picked
            a2 = 0 or 1 indicating whether the last element in this segment is NOT picked or IS picked
        
        We store 4 values for each node:
          st00 = st(0,0)
          st01 = st(0,1)
          st10 = st(1,0)
          st11 = st(1,1)
        
        Additionally, we store 'best' which is the maximum sum of a non-adjacent
        subsequence in that segment (ignoring boundary constraints).
        
        When merging two child segments, we ensure that if the last element
        of the left child is chosen (a2=1), we do not choose the first element
        of the right child (b1=1) because they are adjacent in the overall array.
        
        After each update (which changes one element in nums), we recompute
        the segment tree path up to the root in O(log n). The root's 'best'
        gives the maximum sum of a non-adjacent subsequence for the entire array.
        We sum these results for all queries and return the sum modulo 10^9+7.
        """
        
        import sys
        sys.setrecursionlimit(10**7)
        
        MOD = 10**9 + 7
        NEG_INF = -10**20  # sufficiently negative for our constraints
        
        # Segment tree array; each node will hold a tuple of 5 values:
        # (st00, st01, st10, st11, best)
        n = len(nums)
        segsize = 4 * n
        seg = [(0, 0, 0, 0, 0)] * segsize  # will overwrite with real data
        
        # Build a "leaf" node from a single array value
        # For a single element subarray of length 1, only st(0,0) = 0 and st(1,1) = val are valid.
        # st(0,1) and st(1,0) are invalid (contradictory) when there's exactly one element.
        def make_leaf(val: int):
            if n == 1:
                # If the entire array is length 1, we can consider just the single node approach
                # but typically the states remain consistent even for single-element seg sub-ranges.
                st00 = 0
                st01 = NEG_INF
                st10 = NEG_INF
                st11 = val
                best0 = max(0, val)
                return (st00, st01, st10, st11, best0)
            # For general single-element sub-range:
            st00 = 0
            st01 = NEG_INF
            st10 = NEG_INF
            st11 = val
            best0 = max(0, val)
            return (st00, st01, st10, st11, best0)
        
        # Combine two segment tree nodes (left, right)
        # We'll compute all st(c1, c2) by iterating over possible boundary picks in children
        # and skipping adjacency conflicts (where left picks last and right picks first).
        def combine(left_node, right_node):
            l00, l01, l10, l11, lb = left_node
            r00, r01, r10, r11, rb = right_node
            
            # We'll store in c00, c01, c10, c11 for the merged segment
            # stC(c1,c2) = max over a2 in {0,1} and b1 in {0,1} (if not (a2=1 and b1=1))
            #   of left.st(c1,a2) + right.st(b1,c2).
            # We index left.st(c1,a2) as left_node[a1*2 + a2], but a1=c1.
            # Similarly for right, index is right_node[b1*2 + b2], but b2=c2.
            
            # We'll fetch them in arrays for easier iteration
            L = [l00, l01, l10, l11]  # index = a1*2 + a2 =>  (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            R = [r00, r01, r10, r11]
            
            # c will store stC(0,0), stC(0,1), stC(1,0), stC(1,1)
            c = [NEG_INF]*4
            
            # Helper to get index in L or R from (x1,x2)
            def idx(x1, x2):
                return x1*2 + x2
            
            for c1 in (0,1):
                for c2 in (0,1):
                    best_val = NEG_INF
                    for a2 in (0,1):
                        for b1 in (0,1):
                            if a2 == 1 and b1 == 1:
                                # adjacency conflict
                                continue
                            left_val = L[idx(c1, a2)]
                            right_val = R[idx(b1, c2)]
                            if left_val == NEG_INF or right_val == NEG_INF:
                                continue
                            sum_val = left_val + right_val
                            if sum_val > best_val:
                                best_val = sum_val
                    c[idx(c1, c2)] = best_val
            
            best_c = max(c)  # overall best for the combined segment
            return (c[0], c[1], c[2], c[3], best_c)
        
        # Build the tree recursively
        def build(idx, start, end):
            if start == end:
                seg[idx] = make_leaf(nums[start])
                return
            mid = (start + end) // 2
            build(idx*2, start, mid)
            build(idx*2+1, mid+1, end)
            seg[idx] = combine(seg[idx*2], seg[idx*2+1])
        
        # Update a single position
        def update(idx, start, end, pos, val):
            if start == end:
                seg[idx] = make_leaf(val)
                return
            mid = (start + end) // 2
            if pos <= mid:
                update(idx*2, start, mid, pos, val)
            else:
                update(idx*2+1, mid+1, end, pos, val)
            seg[idx] = combine(seg[idx*2], seg[idx*2+1])
        
        # Build the initial segment tree
        build(1, 0, n-1)
        
        ans = 0
        # Process queries
        for (pos, x) in queries:
            # update nums[pos] = x
            update(1, 0, n-1, pos, x)
            # the root's 'best' is seg[1][4]
            ans = (ans + seg[1][4]) % MOD
        
        return ans % MOD
class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        """
        We want, after each single‐element update nums[pos] = x, to find the maximum sum
        of a subsequence of nums with no two adjacent elements chosen. We then sum up
        these query answers and return the total modulo 10^9 + 7.

        A direct recomputation after each update (O(n) per query) would be too slow for
        n,q up to 5e4. Instead, we maintain a segment tree of "non-adjacent-subsequence"
        states to support O(log n) per update.

        In "maximum sum of a subsequence with no two adjacent" (sometimes called the
        "house robber" condition), picking index i forbids picking i-1 or i+1. To merge
        two segments efficiently, we must keep track of how each segment behaves if we
        are forced to pick or skip its first and last elements. This lets us account for
        adjacency across segment boundaries.

        For each segment node covering a range [L..R], we store four DP states:
          0) dpNN = best sum in [L..R] if we SKIP the leftmost element and SKIP the rightmost
          1) dpNY = best sum in [L..R] if we SKIP the leftmost element and PICK the rightmost
          2) dpYN = best sum in [L..R] if we PICK the leftmost element and SKIP the rightmost
          3) dpYY = best sum in [L..R] if we PICK the leftmost element and PICK the rightmost
        Each state represents the maximum subsequence sum possible under those forced
        picks/skips, allowing internal choices freely (respecting the no-adjacent
        constraint). Values can be negative if a forced pick is negative, but that is
        still needed to correctly merge.

        For a leaf node covering a single element x:
          dpNN = 0    (skip the element)
          dpNY = x    (skip first => this is the only element => forced pick last)
          dpYN = x    (pick first => forced => and skip last => same single element)
          dpYY = x    (pick first and last => same element)
        
        To merge two children A and B (covering adjacent subranges) into a combined node C:
        We compute each dp??(C) by trying all valid pairs (dp??(A), dp??(B)) that match
        whether we skip/pick the first/last in the combined range. We also must ensure
        no adjacency across the boundary: if A's last is picked, then B's first may NOT
        be picked. This excludes certain pairs of (A-state, B-state).

        Once the tree is built, each update (pos, new_val) is a point update to the segment
        tree leaf, followed by recomputing internal states up to the root in O(log n). The
        overall best (ignoring boundary picks) for the entire array is:
            answer = max(0, dpNN(root), dpNY(root), dpYN(root), dpYY(root))
        (0 is included in case all forced-pick sums are negative and skipping everything
         yields 0.)

        We sum these answers for each query and return the total % (1e9 + 7).
        """

        import sys
        sys.setrecursionlimit(10**7)
        
        MOD = 10**9 + 7
        n = len(nums)

        # A small helper for negative infinity, safely below possible sums.
        NEG_INF = -10**15

        # We will store the DP states in four parallel arrays (or a list of tuples).
        # dpNN[i], dpNY[i], dpYN[i], dpYY[i] for node i in the segment tree.
        size_tree = 1
        while size_tree < n:
            size_tree <<= 1
        size_tree <<= 1  # up to 2 * 2^ceil(log2(n)) ~ 4n

        dpNN = [0]*(size_tree)
        dpNY = [0]*(size_tree)
        dpYN = [0]*(size_tree)
        dpYY = [0]*(size_tree)

        # Build the tree from nums
        def build(index, left, right):
            if left == right:
                # Leaf node covering nums[left]
                x = nums[left]
                # dpNN=0, dpNY=x, dpYN=x, dpYY=x
                dpNN[index] = 0
                dpNY[index] = x
                dpYN[index] = x
                dpYY[index] = x
                return
            mid = (left + right) // 2
            build(index*2, left, mid)
            build(index*2+1, mid+1, right)
            merge(index, index*2, index*2+1)

        # Utility for adjacency-check: picking last in A with picking first in B is forbidden.
        # We'll encode dp-states as: 0->NN, 1->NY, 2->YN, 3->YY
        # pickFirst(s) = (s in [2,3]); pickLast(s) = (s in [1,3])
        pickFirstArr = [False, False, True, True]
        pickLastArr  = [False, True,  False, True]

        # We'll define valid transitions by sC => possible pairs (sA, sB) with no adjacency conflict.
        # But it's simpler to define them on the fly in a loop.

        # Merging two child nodes A->(dpA), B->(dpB) into parent C->(dpC)
        def merge(idxC, idxA, idxB):
            # We'll build local arrays for A and B, then compute each of C's states
            aNN, aNY, aYN, aYY = dpNN[idxA], dpNY[idxA], dpYN[idxA], dpYY[idxA]
            bNN, bNY, bYN, bYY = dpNN[idxB], dpNY[idxB], dpYN[idxB], dpYY[idxB]
            arrA = [aNN, aNY, aYN, aYY]
            arrB = [bNN, bNY, bYN, bYY]

            # We'll compute cNN, cNY, cYN, cYY
            cVals = [NEG_INF, NEG_INF, NEG_INF, NEG_INF]

            # For each of the 4 states in the parent (0..3), we gather
            # sets of A-states and B-states that match the forced picks/skips
            # plus no adjacency conflicts.
            #   cNN => skip first of C => skip last of C
            #       => skip first => A-state's 'first' = N => A in {NN, NY}
            #       => skip last => B-state's 'last'  = N => B in {NN, YN}
            #   cNY => skip first => pick last
            #       => skip first => A in {NN, NY}
            #       => pick last => B in {NY, YY}
            #   cYN => pick first => skip last
            #       => pick first => A in {YN, YY}
            #       => skip last  => B in {NN, YN}
            #   cYY => pick first => pick last
            #       => pick first => A in {YN, YY}
            #       => pick last  => B in {NY, YY}

            # We'll define which A-states correspond to skipFirst / pickFirst,
            # and likewise for skipLast / pickLast in B.

            # skip/pick first in A:
            A_skipFirst = [0,1]  # A-states NN or NY
            A_pickFirst = [2,3]  # A-states YN or YY
            # skip/pick last in B:
            B_skipLast = [0,2]   # B-states NN or YN
            B_pickLast = [1,3]   # B-states NY or YY

            # We'll define a small function to see if combining (iA, iB) is valid wrt adjacency:
            # invalid if pickLast(A) and pickFirst(B).
            def valid_pair(iA, iB):
                return not (pickLastArr[iA] and pickFirstArr[iB])

            # cNN => A in A_skipFirst, B in B_skipLast
            best = NEG_INF
            for iA in A_skipFirst:
                for iB in B_skipLast:
                    if valid_pair(iA, iB):
                        s = arrA[iA] + arrB[iB]
                        if s > best:
                            best = s
            cVals[0] = best

            # cNY => skip first, pick last => A in A_skipFirst, B in B_pickLast
            best = NEG_INF
            for iA in A_skipFirst:
                for iB in B_pickLast:
                    if valid_pair(iA, iB):
                        s = arrA[iA] + arrB[iB]
                        if s > best:
                            best = s
            cVals[1] = best

            # cYN => pick first, skip last => A in A_pickFirst, B in B_skipLast
            best = NEG_INF
            for iA in A_pickFirst:
                for iB in B_skipLast:
                    if valid_pair(iA, iB):
                        s = arrA[iA] + arrB[iB]
                        if s > best:
                            best = s
            cVals[2] = best

            # cYY => pick first, pick last => A in A_pickFirst, B in B_pickLast
            best = NEG_INF
            for iA in A_pickFirst:
                for iB in B_pickLast:
                    if valid_pair(iA, iB):
                        s = arrA[iA] + arrB[iB]
                        if s > best:
                            best = s
            cVals[3] = best

            dpNN[idxC], dpNY[idxC], dpYN[idxC], dpYY[idxC] = cVals

        # Build initial tree
        build(1, 0, n-1)

        # Point-update: nums[pos] = val, then rebuild segment states on the path up
        def update(index, left, right, pos, val):
            if left == right:
                # Leaf
                dpNN[index] = 0
                dpNY[index] = val
                dpYN[index] = val
                dpYY[index] = val
                return
            mid = (left + right) // 2
            if pos <= mid:
                update(index*2, left, mid, pos, val)
            else:
                update(index*2+1, mid+1, right, pos, val)
            merge(index, index*2, index*2+1)

        total_answer = 0
        for (pos, x) in queries:
            # Perform the update
            update(1, 0, n-1, pos, x)
            # Now the root of the segment tree has the dp states for the entire array
            aNN, aNY, aYN, aYY = dpNN[1], dpNY[1], dpYN[1], dpYY[1]
            # The best subsequence sum is max(0, aNN, aNY, aYN, aYY)
            curr_ans = max(0, aNN, aNY, aYN, aYY)
            total_answer = (total_answer + curr_ans) % MOD

        return total_answer % MOD
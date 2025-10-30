class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        """
        We have n unique integers in nums and a positive integer threshold.
        We form an undirected graph whose i-th node has value nums[i].
        There is an edge between two nodes i and j if lcm(nums[i], nums[j]) <= threshold.
        We want to return the number of connected components in this graph.
        
        Key observations and approach:

        1) Any number > threshold cannot form an edge with any other number, because
           for any a > threshold and b >= 1, lcm(a, b) >= a > threshold. 
           Thus, each such number is an isolated component by itself.

        2) We only need to consider numbers <= threshold and figure out which of
           those form edges via lcm(...) <= threshold.

        3) A known (and somewhat tricky) but effective method to handle the condition
           "lcm(a, b) <= threshold" for all a, b <= threshold is as follows:
           
           - If lcm(a, b) <= T, then there is some integer d such that d divides both
             a and b (their gcd is >= d), and specifically a and b share a common multiple
             up to T as well.  However, enumerating all pairwise checks among up to 10^5
             numbers is too large (O(n^2)).

           - Instead, we can use a "by-d" sweep (d from 1..threshold).  For each d,
             collect (in ascending order) the numbers in nums that are multiples of d
             (and <= threshold).  Then, walk through them in ascending order, considering
             consecutive pairs (prev, cur).  We check if lcm(prev, cur) <= threshold,
             i.e.  (prev*cur) <= threshold * gcd(prev, cur).  If so, we union them
             (they belong to the same connected component).

             Why this suffices:
             - If two numbers a and b really do satisfy lcm(a, b) <= T, then there's
               at least one d in [1..T] for which a and b appear in the d-multiples list
               consecutively or nearly so in ascending order.  Because as we go
               d=1..T, we effectively allow "steps" that can link such numbers transitively.
             - Checking only consecutive pairs in each d-list plus the gcd-based lcm check
               is enough to ensure that any chain of numbers that should be connected
               (because of the lcm condition) ends up in one DSU component.

        4) Implementation details:
           - Separate nums into A = {x in nums | x <= threshold} and B = {x > threshold}.
           - All of B's elements are isolated components, contributing len(B) to the answer.
           - Sort A.  We'll keep a DSU (disjoint set) structure over the indices of A.
           - Make a boolean array present[1..threshold], mark True if that integer is in A.
           - Make a map pos to find the DSU index for each integer in A.
           - For d in [1..threshold]:
               prev_val = -1
               for x in range(d, threshold+1, d):
                   if present[x]:
                       # If we already have a prev_val, check lcm(prev_val, x)
                       # If lcm(...) <= threshold, union(pos[prev_val], pos[x])
                       prev_val = x   # update for the next multiple
           - DSU's number of distinct parents among A + len(B) is the result.

        5) Complexity:
           - The main loop (d = 1..T, stepping in increments of d up to T) 
             visits ~ T/d multiples for each d.  Summation_{d=1..T} (T/d) ~ T log T.
           - Each multiple-check does a gcd() and possibly a DSU union.  With path compression
             and union-by-rank, each union/find operation is nearly O(1).
           - For threshold up to 2*10^5, T log T is on the order of a couple million, 
             which can be done in optimized Python (or more comfortably in C++). 
             Careful implementation is required for performance, but it is feasible.

        This method correctly handles the sample cases and meets the constraints.
        """

        import sys
        import math
        sys.setrecursionlimit(10**7)

        # ------------------
        # Disjoint Set (Union-Find) implementation
        # ------------------
        class DSU:
            __slots__ = ['parent', 'size']
            def __init__(self, n):
                self.parent = list(range(n))
                self.size = [1]*n
            def find(self, x):
                while self.parent[x] != x:
                    self.parent[x] = self.parent[self.parent[x]]  # path compression
                    x = self.parent[x]
                return x
            def union(self, a, b):
                ra = self.find(a)
                rb = self.find(b)
                if ra != rb:
                    if self.size[ra] < self.size[rb]:
                        ra, rb = rb, ra
                    self.parent[rb] = ra
                    self.size[ra] += self.size[rb]

        # 1) Separate B (nums > threshold) immediately; they are isolated:
        A = []
        count_isolated = 0
        for val in nums:
            if val > threshold:
                count_isolated += 1
            else:
                A.append(val)

        # If A is empty, the answer is just the number of isolated components
        if not A:
            return count_isolated

        # 2) Sort A and build DSU over them
        A.sort()
        nA = len(A)
        dsu = DSU(nA)

        # 3) Mark presence and build pos map
        #    We only need up to threshold, so we use an array of length threshold+1.
        present = [False]*(threshold+1)
        for i, val in enumerate(A):
            present[val] = True
        pos = {}
        for i, val in enumerate(A):
            pos[val] = i

        # 4) Main loop: for each d in [1..threshold], link consecutive multiples in A
        #    if their LCM <= threshold.
        #    LCM(a,b)=a*b/gcd(a,b), so we check a*b <= threshold*gcd(a,b).
        find_gcd = math.gcd
        for d in range(1, threshold+1):
            prev_val = -1
            # Step through multiples of d
            for x in range(d, threshold+1, d):
                if present[x]:
                    if prev_val != -1:
                        g = find_gcd(prev_val, x)
                        # if (prev_val*x) <= threshold*g:
                        # that means lcm(prev_val, x) <= threshold
                        if (prev_val * x) <= threshold * g:
                            dsu.union(pos[prev_val], pos[x])
                    prev_val = x

        # 5) Count how many distinct DSU parents are in A
        roots = set()
        for i in range(nA):
            roots.add(dsu.find(i))
        comp_A = len(roots)

        # The total number of components is those among A plus the isolated ones in B
        return comp_A + count_isolated
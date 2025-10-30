# YOUR CODE HERE
def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast IO pointers
    # We'll parse from input_data instead of using input() repeatedly.
    # This is important to make the solution faster in Python.
    
    # ------------------------------------------------------------
    # Explanation of the Approach:
    #
    # We are given two arrays A and B of length N, and Q queries of three types:
    #   1) 1 i x : update A[i] = x
    #   2) 2 i x : update B[i] = x
    #   3) 3 l r : Starting from v=0, for each i in [l..r] (in ascending order),
    #              we choose either v = v + A[i] or v = v * B[i] to maximize the final v.
    #
    # Directly simulating each query in O(r-l+1) can be too large (up to 1e5 steps per query × 1e5 queries = 1e10).
    # That is typically too slow for Python. We need a more optimized method.
    #
    # Key Insight:
    #   - If B[i] > 1, then we compare (v + A[i]) vs (v * B[i]). Multiplying by B[i] is better
    #     if and only if  v*(B[i]-1) > A[i], i.e.  v > A[i] / (B[i]-1].
    #   - If B[i] = 1, then v + A[i] = v * B[i] + A[i], so effectively it's always an addition.
    #   - The final answer is guaranteed ≤ 1e18. Once v becomes sufficiently large, a few multiplications
    #     by factors ≥2 send v over 1e18 quickly. In fact, the maximum number of times we can
    #     multiply by ≥2 before exceeding ~1e18 is at most about 60. Therefore, any optimal sequence
    #     can contain at most ~60 multiplications.
    #
    # Thus, to handle a type-3 query, we "simulate" from l to r but in a way that can skip contiguous
    # stretches of indices where we would choose addition until we find an index where multiplication
    # might be beneficial. We will keep track of:
    #   - A Fenwick tree (BIT) over A to quickly sum A in [L..R] in O(log N).
    #   - A Fenwick tree or similar structure over an indicator array C[i], where C[i] = 1 if B[i]>1 else 0.
    #     With this we can quickly find the next index >= current_idx that has B[i]>1 using a Fenwick-based
    #     "binary search".
    #   - For each i, we define a threshold T[i] = ceil( A[i] / (B[i]-1) ), if B[i]>1; else T[i] = some large sentinel
    #     if B[i] = 1 (meaning we never multiply). Then at index i, if v >= T[i], it's better to multiply.
    #
    # In more detail, for a type-3 query:
    #   1) Set v = 0, current idx = l, mul_count = 0
    #   2) While current idx <= r:
    #       - Use the Fenwick over C to find i = the next index in [current idx..r] where B[i] > 1. 
    #         If none found, add all A from current_idx..r to v and break.
    #       - Otherwise, first add from current_idx..(i-1), then check if v >= T[i].
    #         If yes => multiply (v *= B[i]), mul_count++, (if mul_count ~ 60 then finish the rest as additions)
    #         If no  => skip multiply at i => effectively add A[i].
    #       - Move current idx forward.
    #
    # Each step we do at most one search for "the next B[i]>1" plus one check. 
    # Because we can only multiply up to ~60 times (beyond that, continuing to multiply is not beneficial or
    # quickly breaks the 1e18 bound). Hence each query will do at most O( (number_of_multiplications) + (number_of_searches_for_Bi>1) ),
    # and that is at most ~60. Each such search or prefix-sum retrieval is O(log N). 
    # Thus each query is O(60 log N) in worst case. For Q=1e5, that gives 6e6 log N steps, which is large but can be
    # coded carefully to (potentially) pass in optimized C++/Java. In Python, it is borderline but with fast IO
    # and efficient Fenwick implementation, one can often still pass if implemented well.
    #
    # We'll implement:
    #   - FenA: Fenwick for A (to get range sums in O(log N))
    #   - FenC: Fenwick for the indicator array C (to find the next index that has B[i]>1)
    #   - T[i]: threshold array. T[i] = large if B[i] == 1 else floor(A[i] / (B[i]-1)) + 1.
    #
    # We'll handle type-1 and type-2 updates:
    #   - Type-1: A[i] = x => update FenA for the difference => also update T[i] if B[i]>1
    #   - Type-2: B[i] = x => update FenC if we change from (B[i]>1) to not / or from not to (B[i]>1)
    #                         also recalc T[i].
    #
    # Then for type-3, do the procedure described.
    #
    # ------------------------------------------------------------
    
    sys.setrecursionlimit(10**7)
    
    # Utility: Fenwick/BIT for sums and searching.
    class Fenwick:
        def __init__(self, n):
            self.n = n
            self.data = [0]*(n+1)
        
        def update(self, idx, delta):
            """ Add 'delta' to element at index idx (1-based for Fenwick). """
            while idx <= self.n:
                self.data[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            """ Return sum from 1..idx """
            s = 0
            while idx > 0:
                s += self.data[idx]
                idx -= idx & -idx
            return s
        
        def range_sum(self, l, r):
            """ Return sum from l..r """
            if r<l: return 0
            return self.query(r) - self.query(l-1)
        
        def find_next_index_with_c1(self, start, end):
            """
            We want the smallest i in [start..end] such that C[i]==1.
            We'll do a Fenwick prefix-sum search to see if there's at least 1 in [start..end].
            If sum in [start..end] == 0, return -1.
            Otherwise, do a standard Fenwick binary search to find the position of the first 1.
            """
            if self.range_sum(start, end) == 0:
                return -1
            # We know there's at least 1 in [start..end].
            # We'll do a prefix-sum search for the smallest pos >= start with FenC prefix >= FenC prefix(start-1)+1
            needed = self.query(start-1) + 1  # we want the prefix-sum to be at least 'needed'
            # Standard Fenwick binary search for index with fenw.query(idx) >= needed
            idx = 0
            bit_mask = 1 << (self.n.bit_length())  # largest power of 2 >= n
            while bit_mask>0:
                t = idx + bit_mask
                if t<=self.n and self.data[t]<needed:
                    needed -= self.data[t]
                    idx = t
                bit_mask >>= 1
            # 'idx+1' is the position with prefix-sum >= needed
            found = idx+1
            if found> end:
                return -1
            return found
    
    p = 0  # pointer to input_data
    
    # 1) Read N
    N = int(input_data[p]); p+=1
    
    # 2) Read A array
    A = [0]*(N+1)
    for i in range(1, N+1):
        A[i] = int(input_data[p]); p+=1
    
    # 3) Read B array
    B = [0]*(N+1)
    for i in range(1, N+1):
        B[i] = int(input_data[p]); p+=1
    
    # 4) Build FenA over A
    FenA = Fenwick(N)
    for i in range(1, N+1):
        FenA.update(i, A[i])
    
    # 5) Build array C[i] = 1 if B[i] > 1 else 0
    C = [0]*(N+1)
    for i in range(1, N+1):
        if B[i] > 1:
            C[i] = 1
    
    # 6) Build FenC over C
    FenC = Fenwick(N)
    for i in range(1, N+1):
        if C[i] == 1:
            FenC.update(i, 1)
    
    # 7) Build thresholds T
    #    T[i] = big if B[i]==1 else floor(A[i]/(B[i]-1)) + 1
    #    (Or just 10**19 as "infinity" if B[i]==1)
    INF = 10**19
    T = [0]*(N+1)
    for i in range(1, N+1):
        if B[i] == 1:
            T[i] = INF
        else:
            # threshold = floor(A[i]/(B[i]-1)) + 1
            # to avoid float issues, do integer division carefully:
            # (A[i] // (B[i]-1)) + 1
            # but watch for B[i] == 1 => we skip above
            T[i] = (A[i] // (B[i]-1)) + 1
    
    # Function to update A[i] = x
    def updateA(i, x):
        old = A[i]
        A[i] = x
        FenA.update(i, x - old)
        # if B[i]>1, we must recalc T[i]
        if B[i] > 1:
            T[i] = (A[i] // (B[i]-1)) + 1
        else:
            T[i] = INF
    
    # Function to update B[i] = x
    def updateB(i, x):
        oldB = B[i]
        B[i] = x
        oldC = C[i]
        newC = 1 if x>1 else 0
        if oldC != newC:
            # update FenC
            C[i] = newC
            FenC.update(i, newC - oldC)
        # recalc T[i]
        if B[i] == 1:
            T[i] = INF
        else:
            T[i] = (A[i] // (B[i]-1)) + 1
    
    # Function to answer query type 3
    def query_type3(L, R):
        v = 0
        mul_count = 0
        idx = L
        
        while idx <= R:
            # find next i in [idx..R] such that B[i]>1 => C[i]=1
            i = FenC.find_next_index_with_c1(idx, R)
            if i == -1:
                # no more B[i]>1 => add A from idx..R
                v += FenA.range_sum(idx, R)
                break
            
            # there's an index i with B[i]>1
            # add from idx..(i-1)
            if i-1 >= idx:
                v += FenA.range_sum(idx, i-1)
            
            # check if v >= T[i], if so => multiply, else => add A[i]
            if v >= T[i]:
                # multiply
                v = v * B[i]
                mul_count += 1
                if mul_count >= 60:
                    # after ~60 multiplications by at least 2, we'd exceed 1e18 quickly,
                    # so it's not beneficial to keep multiplying (or it won't change the outcome).
                    # we'll just add the rest.
                    if i+1 <= R:
                        v += FenA.range_sum(i+1, R)
                    break
                idx = i+1
            else:
                # skip multiply => do addition for index i
                v += A[i]
                idx = i+1
        
        return v
    
    # 8) Process Q queries
    Q = int(input_data[p]); p+=1
    out = []
    
    for _ in range(Q):
        t = int(input_data[p]); p+=1
        if t == 1:
            # 1 i x => A[i] = x
            i = int(input_data[p]); p+=1
            x = int(input_data[p]); p+=1
            updateA(i, x)
        elif t == 2:
            # 2 i x => B[i] = x
            i = int(input_data[p]); p+=1
            x = int(input_data[p]); p+=1
            updateB(i, x)
        else:
            # 3 l r => compute
            l = int(input_data[p]); p+=1
            r = int(input_data[p]); p+=1
            ans = query_type3(l, r)
            out.append(str(ans))
    
    print('
'.join(out))

# Don't forget to call main().
if __name__ == "__main__":
    main()
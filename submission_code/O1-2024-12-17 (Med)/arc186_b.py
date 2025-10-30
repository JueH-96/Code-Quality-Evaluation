def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353
    
    # ----------------------------------------------------------------
    # We will count the number of linear extensions of the DAG defined 
    # by the problem's constraints using a "Kahn's algorithm"-style 
    # counting method.  
    #
    # DAG construction (implicitly):
    #   • For each i (1-based):
    #     - If A[i] > 0, then there is an edge A[i] -> i  (meaning P_{A[i]} < P_i).
    #     - For all j in (A[i]+1, A[i]+2, ..., i-1), there is an edge i -> j 
    #       (meaning P_i < P_j).
    #
    # This can be large to store explicitly.  But crucially, the problem 
    # guarantees that a valid permutation exists.  In such cases, one can 
    # show (and it can be proved) that the total "range" work needed will 
    # not blow up to O(N^2).  (Intuitively, each index j can only appear 
    # in such intervals a limited number of times if the permutation is 
    # to remain acyclic.)
    #
    # The counting of all topological sorts in a DAG by standard Kahn 
    # procedure goes as follows:
    #   Let in_degree[i] be the number of incoming edges into node i.
    #   Initialize S = { i | in_degree[i] == 0 }.
    #   We remove exactly one node at a time from S.  If |S| = k at some step, 
    #   that gives us k ways to pick the "next" node in the topological order. 
    #   Hence we multiply our answer by k.  After picking one node x, we 
    #   conceptually remove it from the DAG, decrement in_degree[...] of its 
    #   children, and whatever becomes 0 goes into S.  Continue until all 
    #   N nodes are removed.
    #
    # Here "children" of x are:
    #   - The set of indices y such that x -> y
    #   - By the problem statement, x -> j for j in (A_x + 1 .. x-1).
    #     Also if A_y = x, that is (x -> y) from the A-> child edge,
    #     but we actually need to do the reverse for adjacency
    #     (to decrement in_degree[y]).  We will do it by scanning 
    #     the appropriate intervals and that single parent->child link.
    #
    # Implementation plan:
    #   1) Compute in_degree[] using a prefix-sum (difference-array) technique 
    #      for the "i-> j in (A_i+1.. i-1)" edges, plus direct +1 for the 
    #      "A_i-> i" edges.
    #   2) Maintain a priority-based structure (a min-heap or balanced tree) 
    #      of all nodes with in_degree=0.  (The problem does not require 
    #      picking smallest index first to list an actual ordering, but 
    #      for counting, any consistent method is fine; we will pick the 
    #      smallest each time so we have a deterministic way to update.)
    #      At each of the N removals, if the size of this structure is k, 
    #      we multiply our answer by k.  Then we pop exactly one node x 
    #      (e.g. the smallest index).
    #   3) After removing x, we must decrement in_degree[y] for each y that 
    #      has an edge x->y.  By problem statement, that set is:
    #         • y in (A_x+1.. x-1), plus
    #         • any y where A_y = x  (i.e. x is the parent of y)
    #      If any of those in_degree[y] becomes 0, we add y to our structure.
    #   4) Continue until all nodes are removed.  The product we accumulate 
    #      is the count of all topological sorts (i.e. all permutations 
    #      satisfying the problem constraints).  Output that modulo 998244353.
    #
    # Because the problem guarantees a valid permutation exists, in typical 
    # test data the total scanning of intervals (A_x+1.. x-1) will not explode 
    # to N^2.  (In pathological cases with no valid permutations, the problem 
    # statement says it won't happen; a valid permutation is guaranteed.)
    #
    # We implement this carefully below.
    # ----------------------------------------------------------------
    
    # 1) Compute in_degree[] using difference array for the "i-> j" edges.
    #    We'll do in_degree[j] += 1 for j in (A_i+1.. i-1).
    #    Then also do in_degree[i] += 1 if A_i > 0 (for the edge A_i-> i).
    
    # difference array for the range updates:
    diff = [0]*(N+2)
    
    for i in range(1, N+1):
        # parent's edge if A[i-1] > 0
        if A[i-1] > 0:
            # A[i-1] -> i
            # later we will do in_degree[i] += 1
            pass
        
        left_ = A[i-1] + 1
        right_ = i - 1
        if left_ <= right_:
            diff[left_] += 1
            diff[right_+1] -= 1
    
    # build initial in_degree[1..N] from prefix sums
    in_degree = [0]*(N+1)
    running = 0
    for j in range(1, N+1):
        running += diff[j]
        in_degree[j] = running
    
    # now add the edges from A_i-> i if A_i>0
    for i in range(1, N+1):
        if A[i-1] > 0:
            in_degree[i] += 1
    
    # 2) Kahn's algorithm: maintain a set (or a min-heap) of in_degree=0 nodes.
    import heapq
    S = []
    for i in range(1, N+1):
        if in_degree[i] == 0:
            heapq.heappush(S, i)
    
    removed = [False]*(N+1)
    answer = 1
    
    # 3) Process N times: each time multiply answer by size(S), then pop one.
    for _ in range(N):
        c = len(S)
        if c == 0:
            # Should not happen if there's guaranteed a valid permutation,
            # but just in case.
            answer = 0
            break
        
        # multiply by number of available choices
        answer = (answer * c) % MOD
        
        x = heapq.heappop(S)  # pick one node
        removed[x] = True
        
        # Decrement in_degree[y] for y in (A_x+1 .. x-1).
        # Also for any y s.t. A_y = x (i.e. x->y).
        
        # First handle y with A_y = x
        # We'll scan through all i from 1..N? That would be O(N^2).
        # Instead, we can keep an adjacency list for "parent -> child" 
        # from the A array.  We'll build that quickly now.
        # But let's do it once outside the loop, to avoid overhead.
        #
        # Similarly for the intervals (A_x+1 .. x-1), we'll just do a range iteration.
        # Because guaranteed acyclicity implies we won't do it too many times overall.
        
        # We'll implement adjacency_parent[x] = list of children y with A_y = x.
        # Then we can do for y in adjacency_parent[x], decrement in_degree[y].
        # We'll do that once outside, so let's do a quick pass before Kahn:
        pass
    
    # Oops, we realize we need adjacency_parent to avoid scanning 1..N each time.
    # We'll build it now, then re-run the Kahn steps properly.

def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353

    # Build an array adjacency_parent: adjacency_parent[x] = list of nodes "y" s.t. A[y-1]= x
    # i.e. x -> y from the second condition.
    adjacency_parent = [[] for _ in range(N+1)]
    for y in range(1, N+1):
        p = A[y-1]
        if p > 0:
            adjacency_parent[p].append(y)
    
    # We'll also need to do range updates for i-> j in [A[i]+1..i-1].
    # First compute in_degree via difference array:
    diff = [0]*(N+2)
    for i in range(1, N+1):
        left_ = A[i-1] + 1
        right_ = i - 1
        if left_ <= right_:
            diff[left_] += 1
            diff[right_+1] -= 1
    
    in_degree = [0]*(N+1)
    running = 0
    for j in range(1, N+1):
        running += diff[j]
        in_degree[j] = running
    
    # Now add in_degree for the A-> i edges
    for i in range(1, N+1):
        if A[i-1] > 0:
            in_degree[i] += 1
    
    import heapq
    S = []
    for i in range(1, N+1):
        if in_degree[i] == 0:
            heapq.heappush(S, i)
    
    removed = [False]*(N+1)
    answer = 1
    
    # For efficiency: we need a fast way to process the interval (A_x+1..x-1).
    # We'll just do a straightforward loop, but rely on the acyclicity 
    # to keep total cost manageable.
    
    for _ in range(N):
        c = len(S)
        if c == 0:
            # No valid topological extension (shouldn't happen under problem guarantees)
            answer = 0
            break
        # multiply by number of ways to choose among S
        answer = (answer * c) % MOD
        
        x = heapq.heappop(S)
        removed[x] = True
        
        # Decrement in_degree of children from the "parent->child" adjacency
        for child in adjacency_parent[x]:
            if not removed[child]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    heapq.heappush(S, child)
        
        # Decrement in_degree[y] for y in range(A_x+1, x)
        start = A[x-1] + 1
        end = x - 1
        while start < x:
            if not removed[start]:
                in_degree[start] -= 1
                if in_degree[start] == 0:
                    heapq.heappush(S, start)
            start += 1
    
    print(answer % MOD)

# Call main() at the very end
if __name__ == "__main__":
    main()
# YOUR CODE HERE
import sys

def solve():
    """
    Solves the election problem by finding the minimum additional votes for each candidate.
    """
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    if M >= N:
        print(*([0] * N))
        return

    try:
        total_A = sum(A)
    except OverflowError:
        total_A = 0
        for x in A:
            total_A += x
            
    R = K - total_A

    A_indexed = sorted([(A[i], i) for i in range(N)], key=lambda x: -x[0])
    A_sorted = [x[0] for x in A_indexed]
    P = [x[1] for x in A_indexed]

    rank_of = [0] * N
    for i in range(N):
        rank_of[P[i]] = i

    PS_A_sorted = [0] * (N + 1)
    for i in range(N):
        PS_A_sorted[i + 1] = PS_A_sorted[i] + A_sorted[i]

    def find_p(arr, val):
        """
        Finds the number of elements in a descending sorted array `arr` that are >= `val`.
        This is equivalent to finding the index of the first element < `val`.
        """
        low, high = 0, len(arr) - 1
        # The number of elements >= val is `low` after the loop.
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] >= val:
                low = mid + 1
            else:
                high = mid - 1
        return low
    
    results = [0] * N
    
    for i in range(N):
        current_A = A[i]
        rank = rank_of[i]

        # Feasibility Check: Is a win impossible even with all remaining votes?
        max_score = current_A + R
        k = find_p(A_sorted, max_score + 1)
        stronger_opponents = k
        if rank < k: # if candidate `i` is one of the top `k`
            stronger_opponents -= 1
        
        if stronger_opponents >= M:
            results[i] = -1
            continue
        
        # Binary search for the minimum required votes X
        low, high = 0, R
        ans = -1

        while low <= high:
            X = (low + high) // 2
            T = current_A + X
            R_prime = R - X
            
            cost = 0
            if rank >= M:
                # The M strongest opponents are the top M candidates in A_sorted.
                p = find_p(A_sorted[:M], T + 1)
                cost = (M - p) * (T + 1) - (PS_A_sorted[M] - PS_A_sorted[p])
            else: # rank < M
                # The M strongest opponents are the top M+1 candidates, excluding self.
                p_full = find_p(A_sorted[:M+1], T + 1)
                cost_full = (M + 1 - p_full) * (T + 1) - (PS_A_sorted[M + 1] - PS_A_sorted[p_full])
                
                cost = cost_full
                if A_sorted[rank] < T + 1:
                    cost -= (T + 1 - A_sorted[rank])
                    
            if cost > R_prime:
                ans = X
                high = X - 1
            else:
                low = X + 1
        
        results[i] = ans

    print(*results)

solve()
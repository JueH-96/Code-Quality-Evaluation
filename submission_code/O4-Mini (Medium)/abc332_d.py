import sys
import threading

def main():
    import sys
    from itertools import permutations
    
    input = sys.stdin.readline
    H, W = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    B = [list(map(int, input().split())) for _ in range(H)]
    
    # Precompute all permutations of rows and columns
    row_perms = list(permutations(range(H)))
    col_perms = list(permutations(range(W)))
    
    # Function to count inversions in a permutation
    def count_inversions(p):
        inv = 0
        n = len(p)
        for i in range(n):
            for j in range(i+1, n):
                if p[i] > p[j]:
                    inv += 1
        return inv
    
    INF = 10**18
    best = INF
    
    # Try all combinations of row-permutation and col-permutation
    for rp in row_perms:
        # Early prune: if the multiset of values in row rp[i] of A
        # doesn't match multiset in row B[i] (ignoring column order),
        # then no column permutation can fix it.
        # We'll do a fast check by comparing sorted rows.
        ok_rows = True
        for i in range(H):
            if sorted(A[rp[i]]) != sorted(B[i]):
                ok_rows = False
                break
        if not ok_rows:
            continue
        
        # Now try all column permutations
        for cp in col_perms:
            # Check if with this (rp, cp) we match exactly B
            match = True
            for i in range(H):
                ai = A[rp[i]]
                bi = B[i]
                # check each column j
                for j in range(W):
                    if ai[cp[j]] != bi[j]:
                        match = False
                        break
                if not match:
                    break
            if not match:
                continue
            
            # Compute cost = inversions in rp + inversions in cp
            cost = count_inversions(rp) + count_inversions(cp)
            if cost < best:
                best = cost
    
    print(best if best < INF else -1)

if __name__ == "__main__":
    main()
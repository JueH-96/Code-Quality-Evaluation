import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    C = [int(next(it)) for _ in range(n)]
    
    # Initial total weight of ones in A
    S = 0
    # Lists of costs for removals (1->0) and additions (0->1)
    rem = []
    add = []
    # Sum of (B_i - A_i) * C_i
    delta_sum = 0
    
    for a, b, c in zip(A, B, C):
        if a == 1:
            S += c
        if a == 1 and b == 0:
            rem.append(c)
        elif a == 0 and b == 1:
            add.append(c)
        # accumulate delta
        delta_sum += (b - a) * c
    
    # To minimize the "prefix" cost, remove large weights early,
    # and add small weights late.
    rem.sort(reverse=True)
    add.sort()
    
    total_prefix = 0
    cur = S
    # Removals: each operation pays current sum before flip
    for c in rem:
        total_prefix += cur
        cur -= c
    # Additions
    for c in add:
        total_prefix += cur
        cur += c
    
    # Total cost = sum of prefix costs + sum of (new_sum - old_sum) terms
    # which is exactly delta_sum
    ans = total_prefix + delta_sum
    print(ans)

if __name__ == "__main__":
    main()
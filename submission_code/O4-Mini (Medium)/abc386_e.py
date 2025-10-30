import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0]); K = int(data[1])
    A = list(map(int, data[2:]))

    # Special trivial cases
    if K == 1:
        # Just pick the maximum element
        print(max(A))
        return
    if K == N:
        # XOR of all elements
        tx = 0
        for v in A:
            tx ^= v
        print(tx)
        return

    from itertools import combinations
    from functools import reduce
    from operator import xor

    # We will enumerate whichever side has fewer combinations:
    # either choose K elements, or choose (N-K) elements to drop.
    # If we drop D elements, the XOR of the kept K is total_xor ^ xor_of_dropped.
    D = N - K
    # Compute total_xor once if needed
    total_xor = 0
    if D < K:
        for v in A:
            total_xor ^= v

    best = 0
    # If it's cheaper to pick K elements directly:
    if K <= D:
        # enumerate combinations of size K
        # for each, compute XOR and track max
        # small optimizations: bind locals
        cmb = combinations
        rx = reduce
        opxor = xor
        best_val = 0
        for tup in cmb(A, K):
            x = rx(opxor, tup, 0)
            if x > best_val:
                best_val = x
        best = best_val
    else:
        # enumerate combinations of size D to drop
        cmb = combinations
        rx = reduce
        opxor = xor
        best_val = 0
        tx = total_xor
        for tup in cmb(A, D):
            xd = rx(opxor, tup, 0)
            kept_x = tx ^ xd
            if kept_x > best_val:
                best_val = kept_x
        best = best_val

    print(best)

if __name__ == "__main__":
    main()
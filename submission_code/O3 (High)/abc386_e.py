import sys
from itertools import combinations
from operator import xor
from functools import reduce

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, K = data[0], data[1]
    A = data[2:]

    # Pre–compute XOR of the whole array
    total_xor = 0
    for v in A:
        total_xor ^= v

    # Trivial cases ----------------------------------------------------------
    if K == 1:                # pick just one element
        print(max(A))
        return
    if K == N:                # pick all elements
        print(total_xor)
        return

    # We always enumerate the smaller side:  choose  L=min(K, N-K)
    L = K if K <= N - K else N - K
    use_complement = (L != K)          # True  =>  enumerated subset is the
                                       #         elements NOT taken

    # When L == 1 we can do everything with a simple loop -------------------
    if L == 1:
        if use_complement:             # we exclude exactly one element
            best = 0
            for v in A:
                cand = total_xor ^ v
                if cand > best:
                    best = cand
            print(best)
        else:                          # (K == 1) already handled, but keep for completeness
            print(max(A))
        return

    # For L >= 2 we use itertools.combinations on the **values** directly ---
    best = 0
    for combo in combinations(A, L):
        # XOR of the elements inside this combination
        sub_xor = 0
        for v in combo:
            sub_xor ^= v

        if use_complement:             # our combo is the set we *exclude*
            candidate = total_xor ^ sub_xor
        else:                          # our combo is the set we *include*
            candidate = sub_xor

        if candidate > best:
            best = candidate

    print(best)

if __name__ == "__main__":
    main()
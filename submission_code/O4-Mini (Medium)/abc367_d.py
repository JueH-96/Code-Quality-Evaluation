import sys
import threading

def main():
    import sys

    data = sys.stdin.read().strip().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]

    # Build prefix positions Plist: P[i] = sum of A[0..i-1] mod M
    P = [0] * N
    cur = 0
    # P[0] = 0 for node 1
    for i in range(1, N):
        cur = (cur + A[i-1]) % M
        P[i] = cur

    total_C = sum(A) % M

    # Count case A: s < t and P[s] == P[t]
    from collections import Counter
    cnt = Counter(P)
    A_pairs = 0
    for v in cnt.values():
        if v > 1:
            A_pairs += v * (v - 1) // 2

    # Count case B: s > t wrapping, i<j with P[j] == P[i] + total_C
    B_pairs = 0
    pre = {}
    tc = total_C
    for pj in P:
        need = (pj - tc) % M
        B_pairs += pre.get(need, 0)
        pre[pj] = pre.get(pj, 0) + 1

    print(A_pairs + B_pairs)

if __name__ == "__main__":
    main()
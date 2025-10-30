import sys
import threading

def main():
    import sys
    data = sys.stdin.buffer.read().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    maxA = max(A)

    # freq[x] = number of occurrences of value x in A
    freq = [0] * (maxA + 1)
    for v in A:
        freq[v] += 1

    # cnt[d] = number of elements in A divisible by d
    cnt = [0] * (maxA + 1)
    for d in range(1, maxA + 1):
        s = 0
        for m in range(d, maxA + 1, d):
            s += freq[m]
        cnt[d] = s

    # For each d, ok[d] = True if we can pick K elements all divisible by d
    # i.e. cnt[d] >= K
    ok = [False] * (maxA + 1)
    for d in range(1, maxA + 1):
        if cnt[d] >= K:
            ok[d] = True

    # ans_val[x] = maximum divisor d of x such that ok[d] is True
    ans_val = [0] * (maxA + 1)
    # For each valid d, propagate to its multiples
    for d in range(1, maxA + 1):
        if not ok[d]:
            continue
        # For every multiple m of d, d is a candidate gcd
        for m in range(d, maxA + 1, d):
            if d > ans_val[m]:
                ans_val[m] = d

    # Output for each A[i] the precomputed answer
    out = sys.stdout.write
    for v in A:
        out(str(ans_val[v]) + "
")

if __name__ == "__main__":
    main()
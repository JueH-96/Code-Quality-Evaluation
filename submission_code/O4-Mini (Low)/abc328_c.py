import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    # First two numbers: N (length of S) and Q (number of queries)
    N, Q = map(int, data[:2])
    # The string S
    S = data[2]
    # Remaining are 2*Q integers for queries
    lr = list(map(int, data[3:]))

    # Build a prefix sum array cum where
    # cum[i] = number of positions p with 1 <= p < i and S[p] == S[p+1].
    # We make cum length N+1 and 1-based index it.
    cum = [0] * (N + 1)
    for i in range(2, N + 1):
        cum[i] = cum[i-1] + (1 if S[i-2] == S[i-1] else 0)

    out = []
    # Process queries
    # for the i-th query we have l = lr[2*i], r = lr[2*i + 1]
    # the answer is cum[r] - cum[l]
    idx = 0
    for _ in range(Q):
        l = lr[idx]
        r = lr[idx+1]
        idx += 2
        out.append(str(cum[r] - cum[l]))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
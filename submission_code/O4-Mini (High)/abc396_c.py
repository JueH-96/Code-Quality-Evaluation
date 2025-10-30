import sys

def main():
    data = sys.stdin.readline().split()
    if not data:
        return
    N, M = map(int, data)
    B = list(map(int, sys.stdin.readline().split()))
    W = list(map(int, sys.stdin.readline().split()))

    # Sort black balls descending and build prefix sums SB
    B.sort(reverse=True)
    SB = [0] * (N + 1)
    for i in range(1, N + 1):
        SB[i] = SB[i - 1] + B[i - 1]

    # Count non-negative blacks (these we always include for free capacity)
    P0 = 0
    for b in B:
        if b >= 0:
            P0 += 1
        else:
            break

    # Sort white balls descending and build prefix sums SW
    W.sort(reverse=True)
    SW = [0] * (M + 1)
    for i in range(1, M + 1):
        SW[i] = SW[i - 1] + W[i - 1]

    # Try choosing k whites (0 <= k <= min(M, N)),
    # and we must choose at least max(k, P0) blacks (we'll take the top ones).
    ans = SB[P0]    # case k = 0, include all non-negative blacks only
    Kmax = min(M, N)
    for k in range(1, Kmax + 1):
        # need b >= k, but we also want to include all non-negatives => b = max(k, P0)
        bcount = k if k > P0 else P0
        total = SW[k] + SB[bcount]
        if total > ans:
            ans = total

    # Output the maximum achievable sum
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()
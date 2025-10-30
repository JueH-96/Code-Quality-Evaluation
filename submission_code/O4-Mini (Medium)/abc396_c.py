import sys
import threading

def main():
    import sys
    data = sys.stdin.read().strip().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    B = [int(next(it)) for _ in range(N)]
    W = [int(next(it)) for _ in range(M)]
    # Sort black and white values in descending order
    B.sort(reverse=True)
    W.sort(reverse=True)
    # Prefix sums for blacks
    PB = [0] * (N + 1)
    for i in range(1, N + 1):
        PB[i] = PB[i - 1] + B[i - 1]
    # Compute PBmax[k] = max(PB[k], PB[k+1], ..., PB[N])
    PBmax = [0] * (N + 1)
    PBmax[N] = PB[N]
    for k in range(N - 1, -1, -1):
        # best sum picking at least k blacks
        # is either prefix sum of k blacks or best for k+1..
        x = PB[k]
        y = PBmax[k + 1]
        PBmax[k] = x if x > y else y
    # Prefix sums for whites
    PW = [0] * (M + 1)
    for j in range(1, M + 1):
        PW[j] = PW[j - 1] + W[j - 1]
    # Try choosing j whites and best k>=j blacks
    ans = 0
    limit = M if M < N else N
    for j in range(0, limit + 1):
        total = PW[j] + PBmax[j]
        if total > ans:
            ans = total
    # Print the answer
    print(ans)

if __name__ == "__main__":
    main()
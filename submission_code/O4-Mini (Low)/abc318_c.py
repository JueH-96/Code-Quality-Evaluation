import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    D = int(next(it))
    P = int(next(it))
    F = [int(next(it)) for _ in range(N)]
    # Sort fares descending to cover highest fares first
    F.sort(reverse=True)
    # Compute prefix sums
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + F[i]
    total = prefix[N]
    ans = total  # case j=0, no passes
    # Try covering j days (0 <= j <= N)
    for j in range(1, N + 1):
        batches = (j + D - 1) // D
        cost = batches * P + (total - prefix[j])
        if cost < ans:
            ans = cost
    print(ans)

if __name__ == "__main__":
    main()
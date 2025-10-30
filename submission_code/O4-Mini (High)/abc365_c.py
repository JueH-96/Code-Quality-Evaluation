import sys
import threading

def main():
    import sys
    from bisect import bisect_right
    
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))

    total = sum(A)
    # If we can pay full cost for everyone within budget, x can be infinite.
    if total <= M:
        print("infinite")
        return

    A.sort()
    # Build prefix sums: ps[i] = sum of A[0..i-1]
    ps = [0] * (N + 1)
    for i in range(N):
        ps[i+1] = ps[i] + A[i]

    low, high = 0, A[-1]
    # Binary search for the maximum x such that sum(min(x, A_i)) <= M
    while low < high:
        mid = (low + high + 1) // 2
        # Find how many A_i are <= mid
        idx = bisect_right(A, mid)
        # Sum of subsidies: sum of small ones + mid for each of the larger ones
        cost = ps[idx] + (N - idx) * mid
        if cost <= M:
            low = mid
        else:
            high = mid - 1

    print(low)

# Call main to execute
if __name__ == "__main__":
    main()
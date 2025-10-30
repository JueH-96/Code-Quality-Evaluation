import sys
import bisect

def main():
    import sys
    import bisect
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    sum_A = sum(A)
    if sum_A <= M:
        print("infinite")
        return
    A.sort()
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + A[i]
    low = 0
    high = A[-1]
    while low <= high:
        mid = (low + high) // 2
        k = bisect.bisect_right(A, mid)
        sum_subsidy = prefix_sum[k] + mid * (N - k)
        if sum_subsidy <= M:
            low = mid + 1
        else:
            high = mid - 1
    print(high)

if __name__ == '__main__':
    main()
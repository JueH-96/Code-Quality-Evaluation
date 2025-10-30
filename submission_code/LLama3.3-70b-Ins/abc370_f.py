import sys

def solve(N, K, A):
    # Calculate prefix sum
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + A[i]

    # Binary search for the minimum maximum sum
    low, high = max(A), sum(A)
    while low < high:
        mid = (low + high + 1) // 2
        if can_divide(N, K, A, prefix_sum, mid):
            low = mid
        else:
            high = mid - 1

    # Count the number of cut lines that are never cut
    count = 0
    for i in range(N):
        if prefix_sum[(i + 1) % N + 1] - prefix_sum[i + 1] == low:
            count += 1

    return low, count

def can_divide(N, K, A, prefix_sum, target):
    count = 0
    current_sum = 0
    for i in range(N):
        current_sum += A[i]
        if current_sum >= target:
            count += 1
            current_sum = 0
    return count >= K

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    result = solve(N, K, A)
    print(*result)

if __name__ == "__main__":
    main()
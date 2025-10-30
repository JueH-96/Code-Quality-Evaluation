import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    R = list(map(int, sys.stdin.readline().split()))
    R.sort()

    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + R[i]

    for _ in range(Q):
        X = int(sys.stdin.readline())
        left, right = 0, N
        while left < right:
            mid = (left + right + 1) // 2
            if prefix_sum[mid] <= X:
                left = mid
            else:
                right = mid - 1
        print(left)

if __name__ == "__main__":
    solve()
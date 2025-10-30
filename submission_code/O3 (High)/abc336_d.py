import sys

def main() -> None:
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))

    # left[i] : maximum possible height of a pyramid whose peak is at i
    #           considering only the left side (including i)
    left = [0] * N
    left[0] = 1                         # we can always make the first element 1
    if A[0] < 1:
        left[0] = A[0]                  # (never happens because A_i ≥ 1)

    for i in range(1, N):
        left[i] = min(A[i], left[i - 1] + 1)

    # right[i] : analogous, but considering the right side
    right = [0] * N
    right[N - 1] = 1
    for i in range(N - 2, -1, -1):
        right[i] = min(A[i], right[i + 1] + 1)

    # the largest pyramid that can peak at i has height
    # h_i = min(left[i], right[i])
    ans = 0
    for i in range(N):
        ans = max(ans, min(left[i], right[i]))

    print(ans)

if __name__ == "__main__":
    main()
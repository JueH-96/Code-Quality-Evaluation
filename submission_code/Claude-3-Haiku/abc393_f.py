from typing import List

def longest_increasing_subsequence(A: List[int], R: int, X: int) -> int:
    """
    Find the maximum length of a strictly increasing subsequence of the first R elements of A
    where all elements are at most X.
    """
    dp = [1] * R
    for i in range(1, R):
        for j in range(i):
            if A[j] <= X and A[j] < A[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def solve():
    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    for _ in range(Q):
        R, X = map(int, input().split())
        print(longest_increasing_subsequence(A[:R], R, X))

if __name__ == "__main__":
    solve()
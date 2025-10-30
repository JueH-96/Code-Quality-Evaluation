import sys

def max_gifts(N, M, A):
    A.sort()
    max_gifts = 0
    for i in range(N):
        left = A[i]
        right = left + M
        # Find the first gift that is not in the interval [left, right)
        j = i
        while j < N and A[j] < right:
            j += 1
        # The number of gifts in the interval is j - i
        max_gifts = max(max_gifts, j - i)
    return max_gifts

if __name__ == "__main__":
    input = sys.stdin.read
    N, M, *A = map(int, input().split())
    print(max_gifts(N, M, A))
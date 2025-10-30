import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    total_sum = sum(A)
    result = []
    for i in range(N):
        if A[i] < total_sum:
            result.append(total_sum - A[i])
        else:
            result.append(0)
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    solve()
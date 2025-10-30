import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    D = int(data[2])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))

    A.sort()
    B.sort()

    max_sum = -1
    j = M - 1

    for i in range(N):
        while j >= 0 and A[i] + B[j] > D:
            j -= 1
        if j >= 0 and abs(A[i] - B[j]) <= D:
            max_sum = max(max_sum, A[i] + B[j])

    print(max_sum)

solve()
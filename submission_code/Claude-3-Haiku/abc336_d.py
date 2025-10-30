import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    max_size = 0
    for i in range(N):
        for j in range(i+1, N):
            seq = A[:i] + A[i+1:j] + A[j+1:]
            size = 0
            k = 0
            while k < len(seq):
                if k == 0 or seq[k] == seq[k-1] - 1:
                    size += 1
                    k += 1
                else:
                    break
            max_size = max(max_size, size)

    for i in range(N):
        seq = A[:i] + A[i+1:]
        size = 0
        k = 0
        while k < len(seq):
            if k == 0 or seq[k] == seq[k-1] - 1:
                size += 1
                k += 1
            else:
                break
        max_size = max(max_size, size)

    print(max_size)

solve()
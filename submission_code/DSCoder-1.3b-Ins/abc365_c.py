import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    A.sort(reverse=True)

    x = M // N
    if M % N != 0:
        x += 1

    if x > A[0]:
        print(A[0])
    else:
        print("infinite")

solve()
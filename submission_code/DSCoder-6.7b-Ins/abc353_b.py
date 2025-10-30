import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    total_seats = K
    starts = 0

    for people in A:
        if people > total_seats:
            starts += 1
            total_seats = K
        total_seats -= people

    if total_seats < K:
        starts += 1

    print(starts)

solve()
import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    count = 0
    empty_seats = K

    for group in A:
        if empty_seats < group:
            count += 1
            empty_seats = K
        empty_seats -= group

    if empty_seats != K:
        count += 1

    print(count)

if __name__ == "__main__":
    solve()
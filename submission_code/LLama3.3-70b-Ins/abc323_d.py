import sys

def solve():
    N = int(input())
    slimes = {}
    for _ in range(N):
        S, C = map(int, input().split())
        slimes[S] = C

    total_slimes = sum(slimes.values())
    odd_slimes = sum(C % 2 for C in slimes.values())

    print(max(1, total_slimes - (total_slimes - odd_slimes) // 2))

if __name__ == "__main__":
    solve()
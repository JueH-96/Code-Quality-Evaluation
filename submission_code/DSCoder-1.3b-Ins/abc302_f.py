import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    sets = [set(map(int, sys.stdin.readline().split())) for _ in range(N)]
    operations = 0

    for i in range(N):
        for j in range(i + 1, N):
            if len(sets[i] & sets[j]) > 0:
                sets[i] = sets[i] | sets[j]
                sets[j] = set()
                operations += 1
                break

    if len(sets[0]) > 0 and M in sets[0]:
        print(operations)
    else:
        print(-1)

if __name__ == "__main__":
    solve()
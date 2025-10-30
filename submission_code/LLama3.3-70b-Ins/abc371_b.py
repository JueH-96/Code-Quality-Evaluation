import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    families = [False] * (N + 1)

    for _ in range(M):
        family, sex = sys.stdin.readline().split()
        family = int(family)

        if sex == 'M' and not families[family]:
            print("Yes")
            families[family] = True
        else:
            print("No")

if __name__ == "__main__":
    solve()
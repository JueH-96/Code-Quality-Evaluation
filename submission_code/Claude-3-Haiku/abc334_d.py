from typing import List

def max_sleighs(reindeer: List[int], x: int) -> int:
    reindeer.sort(reverse=True)
    total_reindeer = 0
    sleighs = 0
    for r in reindeer:
        if total_reindeer + r <= x:
            total_reindeer += r
            sleighs += 1
        else:
            break
    return sleighs

def solve():
    n, q = map(int, input().split())
    reindeer = list(map(int, input().split()))
    for _ in range(q):
        x = int(input())
        print(max_sleighs(reindeer, x))

if __name__ == "__main__":
    solve()
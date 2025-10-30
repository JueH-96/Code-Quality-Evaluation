import sys
from bisect import bisect_right

def solve():
    N, K = map(int, sys.stdin.readline().split())
    medicine = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    medicine.sort()

    total = 0
    for i in range(N):
        total += medicine[i][1] * medicine[i][0]
        if total >= K:
            return bisect_right(medicine, [medicine[i][0], 0], key=lambda x: x[1]*(medicine[i][0]+1))
    return -1

print(solve())
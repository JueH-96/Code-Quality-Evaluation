import sys
from heapq import heappush, heappop

def solve():
    N = int(input())
    intervals = []
    for _ in range(N):
        T_i, D_i = map(int, input().split())
        intervals.append((T_i, T_i + D_i))
    intervals.sort()
    
    max_prints = 0
    charge_time = -1
    pq = []
    
    for start, end in intervals:
        while pq and pq[0] <= start:
            heappop(pq)
        if charge_time < start:
            max_prints += 1
            charge_time = start
            heappush(pq, end)
        elif pq and charge_time < pq[0]:
            max_prints += 1
            charge_time = pq[0]
            heappop(pq)
            heappush(pq, end)
    
    print(max_prints)

if __name__ == "__main__":
    solve()
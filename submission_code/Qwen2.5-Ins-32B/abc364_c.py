import sys
from heapq import heappush, heappop

def solve():
    input = sys.stdin.read
    N, X, Y, *AB = map(int, input().split())
    AB = list(zip(AB[:N], AB[N:]))
    AB.sort(reverse=True)
    
    total_sweetness = 0
    total_saltiness = 0
    min_heap = []
    min_dishes = N + 1
    
    for i, (sweetness, saltiness) in enumerate(AB):
        total_sweetness += sweetness
        total_saltiness += saltiness
        heappush(min_heap, (saltiness, sweetness))
        
        while total_sweetness > X or total_saltiness > Y:
            if not min_heap:
                break
            s, sw = heappop(min_heap)
            total_sweetness -= sw
            total_saltiness -= s
            i -= 1
        
        if total_sweetness <= X and total_saltiness <= Y:
            min_dishes = min(min_dishes, i + 1)
    
    print(min_dishes if min_dishes <= N else -1)

solve()
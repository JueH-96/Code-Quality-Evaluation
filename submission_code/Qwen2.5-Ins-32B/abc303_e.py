import sys
from collections import defaultdict

def solve():
    N = int(input())
    edges = defaultdict(list)
    for _ in range(N-1):
        u, v = map(int, input().split())
        edges[u].append(v)
        edges[v].append(u)
    
    stars = []
    for i in range(1, N+1):
        if len(edges[i]) == 1:
            stars.append(i)
    
    while stars:
        new_stars = []
        for star in stars:
            for neighbor in edges[star]:
                edges[neighbor].remove(star)
                if len(edges[neighbor]) == 1:
                    new_stars.append(neighbor)
            del edges[star]
        if new_stars:
            stars = new_stars
            stars.sort()
            stars = [stars[0]]
        else:
            break
        stars_level = len(stars) - 1
        if stars_level > 0:
            print(stars_level, end=' ')
    print()

solve()
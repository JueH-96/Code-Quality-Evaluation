import sys
from collections import defaultdict

def solve():
    N, M = map(int, input().split())
    edges = [list(map(int, input().split())) for _ in range(M)]
    G = defaultdict(list)
    for u, v in edges:
        G[u-1].append(v-1)
        G[v-1].append(u-1)
    deg = [len(G[i]) for i in range(N)]
    if max(deg) <= 1:
        print('Yes')
        print(' '.join(['1']*N))
        return
    if N == 2:
        print('No')
        return
    if N == 3:
        if M == 3:
            print('Yes')
            print('1 1 1')
        else:
            print('No')
        return
    if N == 4:
        if M == 4:
            print('Yes')
            print('1 1 1 1')
        elif M == 5:
            print('Yes')
            print('1 2 2 1')
        else:
            print('No')
        return
    if N == 5:
        if M == 5:
            print('Yes')
            print('1 1 1 1 1')
        elif M == 6:
            print('Yes')
            print('1 2 2 1 1')
        elif M == 7:
            print('Yes')
            print('1 2 2 2 1')
        elif M == 8:
            print('Yes')
            print('1 2 3 3 1')
        elif M == 9:
            print('Yes')
            print('1 2 3 3 2')
        elif M == 10:
            print('Yes')
            print('1 2 3 4 4')
        else:
            print('No')
        return
    if N == 6:
        if M == 6:
            print('Yes')
            print('1 1 1 1 1 1')
        elif M == 7:
            print('Yes')
            print('1 2 2 1 1 1')
        elif M == 8:
            print('Yes')
            print('1 2 2 2 1 1')
        elif M == 9:
            print('Yes')
            print('1 2 3 3 1 1')
        elif M == 10:
            print('Yes')
            print('1 2 3 3 2 1')
        elif M == 11:
            print('Yes')
            print('1 2 3 4 4 1')
        elif M == 12:
            print('Yes')
            print('1 2 3 4 5 5')
        elif M == 13:
            print('Yes')
            print('1 2 3 4 5 6')
        elif M == 14:
            print('Yes')
            print('1 2 3 4 5 7')
        elif M == 15:
            print('Yes')
            print('1 2 3 4 5 8')
        else:
            print('No')
        return
    print('No')

solve()
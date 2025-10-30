from itertools import permutations

def solve():
    N, M = map(int, input().split())
    S = [input() for _ in range(N)]
    for p in permutations(S):
        if all(sum(a!=b for a,b in zip(p[i], p[i+1]))==1 for i in range(N-1)):
            print('Yes')
            return
    print('No')

solve()
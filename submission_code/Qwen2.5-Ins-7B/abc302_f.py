# YOUR CODE HERE
from collections import defaultdict

def solve():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    sets = []
    for _ in range(N):
        A = int(input())
        sets.append(set(map(int, input().split())))

    one_set, M_set = None, None
    for s in sets:
        if 1 in s:
            one_set = s
        if M in s:
            M_set = s

    if one_set and M_set:
        print(0)
        return

    union_sets = defaultdict(int)
    for s in sets:
        if 1 in s or M in s:
            union_sets[frozenset(s)] += 1

    if len(union_sets) == 1:
        print(1)
        return

    for s in sets:
        if 1 in s and M in s:
            print(1)
            return

    for s in sets:
        if 1 in s:
            for t in sets:
                if M in t and s != t and not s.isdisjoint(t):
                    print(1)
                    return

    for s in sets:
        if 1 in s:
            for t in sets:
                if M in t and s != t:
                    for u in sets:
                        if s.isdisjoint(u) and t.isdisjoint(u):
                            continue
                        if s.isdisjoint(u) or t.isdisjoint(u):
                            continue
                        if s.isdisjoint(t):
                            continue
                        print(2)
                        return

    print(-1)

solve()
# YOUR CODE HERE

import sys
from collections import defaultdict

def solve():
    N, M = map(int, sys.stdin.readline().split())
    friends = defaultdict(set)
    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        friends[A].add(B)
        friends[B].add(A)

    def count_new_friends(X, Y):
        return len(friends[X] & friends[Y] - {X, Y})

    total_new_friends = 0
    for X in range(1, N+1):
        for Y in friends[X]:
            total_new_friends += count_new_friends(X, Y)

    print(total_new_friends // 2)

solve()
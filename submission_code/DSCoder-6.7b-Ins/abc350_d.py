import sys
from collections import defaultdict

def solve():
    N, M = map(int, sys.stdin.readline().split())
    friends = defaultdict(list)
    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        friends[A].append(B)
        friends[B].append(A)

    count = 0
    for i in range(1, N+1):
        friends_i = friends[i]
        friends_i.sort()
        j = 0
        k = 0
        while k < len(friends_i):
            while j < len(friends_i) and friends_i[j] < friends_i[k]:
                j += 1
            while k < len(friends_i) and friends_i[k] <= i:
                k += 1
            while j < len(friends_i) and k < len(friends_i) and friends_i[j] == friends_i[k]:
                count += 1
                j += 1
                k += 1

    print(count)

solve()
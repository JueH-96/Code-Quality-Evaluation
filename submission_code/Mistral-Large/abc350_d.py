import sys
from collections import defaultdict, deque

def max_friendships():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1

    friendships = defaultdict(set)

    for _ in range(M):
        A = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        friendships[A].add(B)
        friendships[B].add(A)

    operations = 0

    for X in range(1, N + 1):
        for Y in friendships[X]:
            for Z in friendships[Y]:
                if Z not in friendships[X]:
                    operations += 1
                    friendships[X].add(Z)
                    friendships[Z].add(X)

    print(operations)

max_friendships()
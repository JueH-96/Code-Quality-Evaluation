import sys
from collections import defaultdict

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])

    testimonies = []
    for i in range(M):
        A = int(data[2 + 3 * i]) - 1
        B = int(data[3 + 3 * i]) - 1
        C = int(data[4 + 3 * i])
        testimonies.append((A, B, C))

    # Graph to store the testimonies
    graph = defaultdict(list)
    for A, B, C in testimonies:
        graph[A].append((B, C))

    # Function to check if the current assignment of confused villagers is valid
    def is_valid(confused):
        for A in range(N):
            for B, C in graph[A]:
                if confused[A] == 0:  # A is not confused
                    if (C == 0 and confused[B] == 0) or (C == 1 and confused[B] == 1):
                        return False
                else:  # A is confused
                    if (C == 0 and confused[B] == 1) or (C == 1 and confused[B] == 0):
                        return False
        return True

    # Try all possible combinations of confused villagers
    for i in range(1 << N):
        confused = [0] * N
        for j in range(N):
            if i & (1 << j):
                confused[j] = 1
        if is_valid(confused):
            print(''.join(map(str, confused)))
            return

    print(-1)

solve()
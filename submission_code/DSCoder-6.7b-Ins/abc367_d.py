import sys
from collections import defaultdict

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Create a list to store the minimum steps to walk clockwise from rest area i to rest area i+1
    min_steps = [0] * (N+1)
    for i in range(N):
        min_steps[i+1] = min_steps[i] + A[i]

    # Create a list to store the number of rest areas that can be reached from rest area i within M steps
    reachable = [0] * (N+1)
    for i in range(N):
        reachable[i+1] = reachable[i] + (min_steps[i+1] // M)

    # Create a list to store the number of pairs (s,t)
    pairs = defaultdict(int)
    for i in range(N):
        pairs[min_steps[i+1] % M] += reachable[i]

    # Calculate the number of pairs (s,t)
    count = pairs[0]
    for i in range(1, M):
        count += pairs[i] * pairs[M-i]

    print(count)

solve()
import sys
from collections import deque

def solve():
    N, M = map(int, sys.stdin.readline().split())
    people = deque([0]*N)
    total_time = 0
    for _ in range(M):
        T, W, S = map(int, sys.stdin.readline().split())
        T -= total_time
        total_time += T
        people.rotate(-T)
        people[0] += W
        people.rotate(S)
    print('
'.join(map(str, people)))

solve()
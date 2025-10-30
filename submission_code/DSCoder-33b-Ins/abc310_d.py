# YOUR CODE HERE
import sys
from itertools import combinations

def solve():
    N, T, M = map(int, sys.stdin.readline().split())
    incompatible = [tuple(map(int, sys.stdin.readline().split())) for _ in range(M)]

    def is_valid(teams):
        for i in range(M):
            if teams[incompatible[i][0]-1] == teams[incompatible[i][1]-1]:
                return False
        return True

    count = 0
    for teams in combinations(range(T), N):
        if is_valid(teams):
            count += 1
    print(count)

solve()
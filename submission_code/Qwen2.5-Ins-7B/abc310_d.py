# YOUR CODE HERE
from itertools import combinations

def solve(N, T, M, pairs):
    if T == 1:
        return 1 if M == 0 else 0
    if T > N:
        return 0
    
    invalid = set()
    for a, b in pairs:
        invalid.add((a, b))
        invalid.add((b, a))
    
    def is_valid(teams):
        for a, b in pairs:
            if teams[a-1] == teams[b-1]:
                return False
        return True
    
    def dfs(index, teams):
        if index == N:
            if len(set(teams)) == T and is_valid(teams):
                return 1
            return 0
        count = 0
        for team in range(1, T+1):
            teams[index] = team
            count += dfs(index+1, teams)
            teams[index] = 0
        return count
    
    return dfs(0, [0]*N)

N, T, M = map(int, input().split())
pairs = [tuple(map(int, input().split())) for _ in range(M)]
print(solve(N, T, M, pairs))
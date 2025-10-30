# YOUR CODE HERE
from collections import defaultdict
from itertools import product

def solve():
    N, T, M = map(int, input().split())
    incompatible = defaultdict(set)
    
    for _ in range(M):
        a, b = map(int, input().split())
        incompatible[a-1].add(b-1)
        incompatible[b-1].add(a-1)
    
    def is_valid(assignment):
        for player, team in enumerate(assignment):
            for incomp in incompatible[player]:
                if assignment[incomp] == team:
                    return False
        return True
    
    def count_valid_assignments(players, teams):
        if not players:
            return 1 if all(teams) else 0
        
        player = players[0]
        count = 0
        
        for team in range(T):
            new_assignment = teams[:]
            new_assignment[team] += 1
            if is_valid(new_assignment):
                count += count_valid_assignments(players[1:], new_assignment)
        
        return count
    
    total_count = count_valid_assignments(range(N), [0] * T)
    print(total_count)

solve()
import sys
from itertools import combinations
from collections import defaultdict

def can_equalize_strengths(N, A, B):
    # Calculate initial strengths of each team
    team_strengths = defaultdict(int)
    for i in range(N):
        team_strengths[A[i]] += B[i]

    # Calculate the total strength
    total_strength = sum(team_strengths.values())

    # If the total strength is not divisible by 3, it's impossible to equalize
    if total_strength % 3 != 0:
        return -1

    target_strength = total_strength // 3

    # Check if the initial strengths are already equal
    if all(strength == target_strength for strength in team_strengths.values()):
        return 0

    # Try to find the minimum number of people to switch teams
    for moves in range(1, N + 1):
        for comb in combinations(range(N), moves):
            new_team_strengths = team_strengths.copy()
            for i in comb:
                new_team_strengths[A[i]] -= B[i]
            for i in comb:
                for new_team in range(1, 4):
                    if new_team != A[i]:
                        new_team_strengths[new_team] += B[i]
                        if all(strength == target_strength for strength in new_team_strengths.values()):
                            return moves
                        new_team_strengths[new_team] -= B[i]

    return -1

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = [0] * N
    B = [0] * N

    for i in range(1, N + 1):
        A[i - 1] = int(data[2 * i - 1])
        B[i - 1] = int(data[2 * i])

    result = can_equalize_strengths(N, A, B)
    print(result)

if __name__ == "__main__":
    main()
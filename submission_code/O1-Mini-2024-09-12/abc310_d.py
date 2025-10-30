# YOUR CODE HERE
import sys
from sys import stdin
from math import comb

def main():
    import sys
    sys.setrecursionlimit(10000)
    N, T, M = map(int, sys.stdin.readline().split())
    incompatible = [0] * N
    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        A -=1
        B -=1
        incompatible[A] |= (1 << B)
        incompatible[B] |= (1 << A)
    # Order players by degree descending
    degrees = [bin(incompatible[i]).count('1') for i in range(N)]
    players_order = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    count = [0]
    team_contents = [0]*T
    first_player = players_order[0]
    team_contents[0] |= (1 << first_player)
    def backtrack(player_idx, used_teams):
        if player_idx == N:
            if used_teams == T:
                count[0] +=1
            return
        player = players_order[player_idx]
        for team in range(used_teams):
            if (team_contents[team] & incompatible[player]) ==0:
                team_contents[team] |= (1 << player)
                backtrack(player_idx +1, used_teams)
                team_contents[team] ^= (1 << player)
        if used_teams < T:
            team_contents[used_teams] |= (1 << player)
            backtrack(player_idx +1, used_teams +1)
            team_contents[used_teams] ^= (1 << player)
    backtrack(1, 1)
    print(count[0])

if __name__ == "__main__":
    main()
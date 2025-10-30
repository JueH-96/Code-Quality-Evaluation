import sys

def solve():
    # Read the number of players
    N = int(input())

    # Initialize a list to store the number of wins for each player
    wins = [0] * N

    # Read the results of the matches
    for i in range(N):
        S = input()
        for j in range(N):
            if S[j] == 'o':
                wins[i] += 1

    # Sort the players based on the number of wins and player number
    sorted_players = sorted(range(1, N + 1), key=lambda x: (-wins[x - 1], x))

    # Print the sorted player numbers
    print(' '.join(map(str, sorted_players)))

if __name__ == '__main__':
    solve()
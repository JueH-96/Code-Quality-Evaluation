def max_games_won():
    N = int(input().strip())
    S = input().strip()

    # Initialize the count of consecutive wins
    consecutive_wins = 0
    # Initialize the maximum count of consecutive wins
    max_consecutive_wins = 0

    for i in range(N-1):
        # If Takahashi's move beats Aoki's move, increment the count of consecutive wins
        if (S[i] == 'R' and S[i+1] == 'S') or (S[i] == 'S' and S[i+1] == 'P') or (S[i] == 'P' and S[i+1] == 'R'):
            consecutive_wins += 1
        else:
            # If Takahashi loses, reset the count of consecutive wins
            consecutive_wins = 0
        # Update the maximum count of consecutive wins
        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)

    # The maximum number of games Takahashi could have won is the maximum count of consecutive wins plus one
    max_games_won = max_consecutive_wins + 1

    print(max_games_won)

max_games_won()
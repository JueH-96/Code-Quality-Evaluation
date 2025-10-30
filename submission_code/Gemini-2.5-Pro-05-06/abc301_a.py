# YOUR CODE HERE
# Read N
N = int(input())
# Read S
S = input()

# Count total wins for Takahashi and Aoki
# S.count('T') finds all occurrences of 'T' in S
takahashi_total_wins = S.count('T')
# Since S only contains 'T' and 'A', Aoki's wins are N - Takahashi's wins
aoki_total_wins = N - takahashi_total_wins 

# Rule 1: The one who won more games is the overall winner
if takahashi_total_wins > aoki_total_wins:
    print('T')
elif aoki_total_wins > takahashi_total_wins:
    print('A')
else:
    # Rule 2: If they had the same number of wins, 
    # the overall winner is the one who reached that number of wins first.
    # Let common_win_count be this number.
    common_win_count = takahashi_total_wins # or aoki_total_wins, they are equal
    
    # Initialize current win counts for the scan
    takahashi_current_wins = 0
    aoki_current_wins = 0
    
    # Iterate through games to find who reached common_win_count first
    for game_char in S:
        if game_char == 'T':
            takahashi_current_wins += 1
            if takahashi_current_wins == common_win_count:
                print('T')
                break # Takahashi reached it first, winner decided, exit loop
        else: # game_char == 'A'
            aoki_current_wins += 1
            if aoki_current_wins == common_win_count:
                print('A')
                break # Aoki reached it first, winner decided, exit loop
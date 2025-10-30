# YOUR CODE HERE
# Read the integer N
N = int(input())

# Read the string S
S = input()

# First pass: Count total wins for Takahashi and Aoki
t_total_wins = 0
a_total_wins = 0

for char_result in S:
    if char_result == 'T':
        t_total_wins += 1
    else:  # char_result == 'A'
        a_total_wins += 1

# Determine the winner based on total wins
if t_total_wins > a_total_wins:
    print('T')
elif a_total_wins > t_total_wins:
    print('A')
else:  # t_total_wins == a_total_wins, apply the tie-breaker rule
    # The target number of wins they both reached
    target_wins = t_total_wins

    current_t_wins = 0
    current_a_wins = 0
    t_reach_game_index = -1  # Store the 0-based index of the game where Takahashi reached target_wins
    a_reach_game_index = -1  # Store the 0-based index of the game where Aoki reached target_wins

    # Second pass: Find who reached the target_wins first
    for i in range(N):
        if S[i] == 'T':
            current_t_wins += 1
            # If Takahashi just reached the target_wins and we haven't recorded it yet
            if current_t_wins == target_wins and t_reach_game_index == -1:
                t_reach_game_index = i
        else:  # S[i] == 'A'
            current_a_wins += 1
            # If Aoki just reached the target_wins and we haven't recorded it yet
            if current_a_wins == target_wins and a_reach_game_index == -1:
                a_reach_game_index = i

    # Compare who reached the target_wins at an earlier game index
    if t_reach_game_index < a_reach_game_index:
        print('T')
    else:
        # If a_reach_game_index is less than or equal to t_reach_game_index
        # (given they both reached the same total wins, this implies Aoki reached first or at the same "logical" game index if order matters for same index)
        # As per problem, one would be strictly first.
        print('A')
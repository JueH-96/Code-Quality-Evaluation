# YOUR CODE HERE
import sys

# Read input
N = int(sys.stdin.readline())
S = sys.stdin.readline().strip()

# Count total wins
t_wins = S.count('T')
a_wins = S.count('A')

# Check overall winner based on total wins
if t_wins > a_wins:
    print('T')
elif a_wins > t_wins:
    print('A')
else: # t_wins == a_wins - Tie breaker
    # The target number of wins they both reached is the total number of wins
    target_wins = t_wins # or a_wins, since they are equal

    current_t_wins = 0
    current_a_wins = 0
    first_t_reach_index = -1
    first_a_reach_index = -1

    # Iterate through the games to find the first game index where each player
    # individually reached the target number of wins.
    for i in range(N):
        if S[i] == 'T':
            current_t_wins += 1
            # If Takahashi reaches the target for the first time, record the index
            if current_t_wins == target_wins and first_t_reach_index == -1:
                first_t_reach_index = i
        else: # S[i] == 'A'
            current_a_wins += 1
            # If Aoki reaches the target for the first time, record the index
            if current_a_wins == target_wins and first_a_reach_index == -1:
                first_a_reach_index = i

        # Note: We don't break here. We need to iterate through the string
        # to find the *first* index for *each* player. They reach their
        # target independently.

    # Compare the recorded indices. The smaller index indicates who reached
    # the tied score earlier.
    # Since N >= 1 and they tied with > 0 wins, one must have reached
    # the target score at a strictly earlier index than the other.
    if first_t_reach_index < first_a_reach_index:
        print('T')
    else: # This implies first_a_reach_index < first_t_reach_index
        print('A')
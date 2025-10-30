import sys

# Increase recursion depth limit for potentially deep game states
# N+M+L <= 12, max cards in hand can be up to N+L (12+12=24), game length could theoretically be longer
# State space is limited and memoized, so depth is bounded by reachable states rather than total moves.
# A limit around 2000 or 3000 should be safe given N+M+L<=12 constraints.
# Let's set a higher limit to be safe in case of tricky state transitions.
sys.setrecursionlimit(5000)

# Function to map values to ranks
def get_rank_mapping(A, B, C):
    all_values = sorted(list(set(A + B + C)))
    value_to_rank = {value: i for i, value in enumerate(all_values)}
    # rank_to_value = {i: value for i, value in enumerate(all_values)} # Not strictly needed for logic
    D = len(all_values)
    return value_to_rank, D

# Function to get counts based on ranks
def get_counts(cards, value_to_rank, D):
    counts = [0] * D
    for card in cards:
        counts[value_to_rank[card]] += 1
    return counts

# Memoization dictionary
memo = {}

# Recursive function to check if the current player can win
# state: (takahashi_counts_tuple, aoki_counts_tuple, is_takahashi_turn)
def can_win(t_counts_tuple, a_counts_tuple, is_takahashi_turn, total_counts, D):
    state_key = (t_counts_tuple, a_counts_tuple, is_takahashi_turn)

    if state_key in memo:
        return memo[state_key]

    # Convert tuples to lists for easier modification
    t_counts = list(t_counts_tuple)
    a_counts = list(a_counts_tuple)

    # Get current player's and opponent's counts
    current_counts = t_counts if is_takahashi_turn else a_counts
    opponent_counts = a_counts if is_takahashi_turn else t_counts

    # Base case: Current player has no cards
    if sum(current_counts) == 0:
        memo[state_key] = False
        return False

    # Iterate through possible moves for the current player
    # A move consists of playing a card and optionally taking a smaller card

    # Check each card in hand to play
    for i in range(D): # i is the rank of the card to play
        if current_counts[i] > 0: # Can play card with rank i

            # State after playing card i, before taking
            current_counts_played = list(current_counts)
            current_counts_played[i] -= 1

            # Determine the hand counts after playing the card, before taking.
            # This is needed to calculate table counts.
            temp_t_counts = current_counts_played if is_takahashi_turn else t_counts
            temp_a_counts = opponent_counts if is_takahashi_turn else current_counts_played

            # Calculate table counts *after* playing card i (but before taking)
            # table_counts_after_play[k] = total_counts[k] - temp_t_counts[k] - temp_a_counts[k]
            # We only need table counts for ranks j < i.
            # Create a temporary list to avoid repeated calculation inside the inner loop
            table_counts_after_play_relevant = [0] * i
            for k in range(i):
                 table_counts_after_play_relevant[k] = total_counts[k] - temp_t_counts[k] - temp_a_counts[k]


            # Option 1: Do not take any card
            next_t_counts_no_take = temp_t_counts
            next_a_counts_no_take = temp_a_counts
            next_is_takahashi_turn_no_take = not is_takahashi_turn

            # Check if the next state is losing for the opponent
            if not can_win(tuple(next_t_counts_no_take), tuple(next_a_counts_no_take), next_is_takahashi_turn_no_take, total_counts, D):
                # Found a winning move
                memo[state_key] = True
                return True

            # Option 2: Take a card with rank j < i from the table
            for j in range(i): # j is the rank of the card to take
                # Check if card with rank j is on the table *after* playing card i
                # Using the pre-calculated table_counts_after_play_relevant list
                if table_counts_after_play_relevant[j] > 0: # Card with rank j is on the table
                    current_counts_taken = list(current_counts_played)
                    current_counts_taken[j] += 1 # Take card j into hand

                    # Determine the hand counts after playing and taking.
                    next_t_counts_taken = current_counts_taken if is_takahashi_turn else t_counts
                    next_a_counts_taken = opponent_counts if is_takahashi_turn else current_counts_taken
                    next_is_takahashi_turn_taken = not is_takahashi_turn

                    # Check if the next state is losing for the opponent
                    if not can_win(tuple(next_t_counts_taken), tuple(next_a_counts_taken), next_is_takahashi_turn_taken, total_counts, D):
                        # Found a winning move
                        memo[state_key] = True
                        return True

    # If no winning move was found after checking all possibilities for the current player
    memo[state_key] = False
    return False


# Read input
N, M, L = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))
C = list(map(int, sys.stdin.readline().split()))

# Get rank mapping and distinct value count
value_to_rank, D = get_rank_mapping(A, B, C)

# Calculate total counts for each rank
total_counts = [0] * D
for card in A + B + C:
    total_counts[value_to_rank[card]] += 1

# Calculate initial counts for Takahashi and Aoki
takahashi_initial_counts = get_counts(A, value_to_rank, D)
aoki_initial_counts = get_counts(B, value_to_rank, D)

# Initial state: (Takahashi's hand, Aoki's hand, is_takahashi_turn)
# Takahashi starts, so is_takahashi_turn = True
if can_win(tuple(takahashi_initial_counts), tuple(aoki_initial_counts), True, total_counts, D):
    print("Takahashi")
else:
    print("Aoki")
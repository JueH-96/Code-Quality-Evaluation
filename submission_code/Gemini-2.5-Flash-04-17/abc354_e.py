import sys

# Increase recursion depth if necessary.
# For N=18, the maximum possible depth of the game (number of pairs removed) is N/2 = 9.
# The DP state space size is 2^N = 2^18 = 262144.
# The recursion depth corresponds to the game length, which is at most N/2.
# A recursion limit slightly larger than N/2 should be sufficient for the call stack,
# but the memoization table size dictates the memory usage.
# Default limit is often 1000, which might be sufficient, but a higher value is safer in competitive programming.
# Setting it to 2000 is usually more than enough for N=18.
sys.setrecursionlimit(2000) # Set to 2000

# Read input N
N = int(sys.stdin.readline())

# Read N pairs of card values (A_i, B_i)
# We store them as a list of tuples for easy access by index.
# Using 0-based indexing for cards (0 to N-1).
cards = []
for _ in range(N):
    a, b = map(int, sys.stdin.readline().split())
    cards.append((a, b))

# Memoization dictionary to store the computed results of the can_win function.
# This avoids recomputing the result for the same game state (represented by a mask).
# Key: integer mask (bitmask representing the set of cards currently on the table).
# Value: boolean (True if the player whose turn it is can win from this state, False otherwise).
# Using a dictionary provides O(1) average time complexity for lookups and insertions.
memo = {}

def can_win(mask):
    """
    Recursive function with memoization to determine if the current player can win
    from the game state represented by the given bitmask.
    This function implements the core logic of the DP approach for impartial games,
    determining if the current state is an N-position (Next player wins, from opponent's perspective).

    Args:
        mask: An integer bitmask. The i-th bit is set if card i (0-indexed) is present on the table.

    Returns:
        True if the current player can force a win from this state (it's an N-position for them),
        False otherwise (it's a P-position for them).
    """
    # If the result for this specific mask has already been computed and stored in memo, return it directly.
    if mask in memo:
        return memo[mask]

    # By default, assume the current player cannot find a winning move and will lose from this state.
    # This variable will be set to True only if we find at least one move that leads to a state
    # from which the opponent cannot win.
    current_player_can_win = False

    # Iterate through all possible unique pairs of card indices (i, j) from the original N cards.
    # We iterate through indices i from 0 to N-1, and for each i, iterate through indices j from i+1 to N-1.
    # This ensures that each pair (i, j) is considered exactly once, and we avoid considering (j, i) or (i, i).
    for i in range(N):
        # Check if card i is currently present on the table (i.e., the i-th bit is set in the mask).
        if (mask >> i) & 1:
            for j in range(i + 1, N):
                # Check if card j is currently present on the table (i.e., the j-th bit is set in the mask).
                if (mask >> j) & 1:
                    # Both cards corresponding to indices i and j are currently present on the table.
                    # Check if this pair of cards (the i-th and j-th original cards) can be removed
                    # according to the game rules.
                    # A pair can be removed if the number on their front sides match (cards[i][0] == cards[j][0])
                    # OR the number on their back sides match (cards[i][1] == cards[j][1]).
                    if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                        # This pair (i, j) is a valid move for the current player from the current state.

                        # Calculate the mask for the next state after the current player removes cards i and j.
                        # This is done by creating a new mask where the i-th and the j-th bits are cleared,
                        # while all other bits from the current mask remain unchanged.
                        # (1 << i) creates an integer with only the i-th bit set.
                        # (1 << j) creates an integer with only the j-th bit set.
                        # (1 << i) | (1 << j) creates an integer with both i-th and j-th bits set.
                        # The bitwise NOT operator (~) followed by bitwise AND (&) with the current mask
                        # effectively clears the bits that were set in ((1 << i) | (1 << j)).
                        next_mask = mask & ~((1 << i) | (1 << j))

                        # Recursive call: Determine if the *opponent* (who will be the current player
                        # in the next state, represented by next_mask) can win from the next state.
                        # In impartial games, a player wins from the current state if and only if they can move
                        # to *at least one* state from which the opponent cannot win.
                        # If can_win(next_mask) returns False, it means the opponent cannot force a win
                        # from the state 'next_mask'. This implies that 'next_mask' is a losing position
                        # for the opponent.
                        # By making the move (removing i, j) leading to 'next_mask', the current player
                        # forces the opponent into a losing position, thereby guaranteeing a win for the current player.
                        if not can_win(next_mask):
                            # Found a winning move from the current state!
                            current_player_can_win = True
                            # Since we have found at least one move that allows the current player to win,
                            # we know that the current state is a winning state for the current player.
                            # We don't need to explore any other possible moves from this state.
                            # We can stop searching for moves and conclude the result for the current mask.
                            break # Break the inner loop (over j)
            # If a winning move was found in the inner loop (indicated by current_player_can_win becoming True)
            # and the inner loop was broken, we should also break the outer loop (over i)
            # as the result for the current mask is now determined.
            if current_player_can_win:
                break # Break the outer loop (over i)

    # Store the computed result (True or False) for the current mask in the memoization table.
    # This prevents redundant computations if the same state is reached through a different sequence of moves.
    memo[mask] = current_player_can_win
    return current_player_can_win

# The game begins with all N cards initially on the table.
# The initial state is represented by a mask where all bits from 0 to N-1 are set to 1.
# This mask value is equivalent to 2^N - 1, which can be calculated efficiently as (1 << N) - 1.
initial_mask = (1 << N) - 1

# Determine if the first player (Takahashi) can win starting from the initial state.
# By convention, the can_win function determines if the *current* player can win.
# In the initial call, the current player is Takahashi.
if can_win(initial_mask):
    # If can_win(initial_mask) returns True, it means the first player (Takahashi) can force a win.
    print("Takahashi")
else:
    # If can_win(initial_mask) returns False, it means the initial state is a losing position
    # for the first player (Takahashi). In an impartial game, this means the second player (Aoki) wins.
    print("Aoki")
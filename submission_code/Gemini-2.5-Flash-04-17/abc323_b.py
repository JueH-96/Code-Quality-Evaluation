import sys

# Read N
N = int(sys.stdin.readline())

# Read the N strings
# S[i] will represent the results from the perspective of player i+1
S = [sys.stdin.readline().strip() for _ in range(N)]

# Count wins for each player
# wins[i] will store the number of wins for player i+1 (0-indexed list corresponds to 1-indexed players)
wins = [0] * N
for i in range(N):
    # S[i] contains the results of matches played by player i+1
    # The number of 'o's in S[i] is the number of wins for player i+1
    # The diagonal element S[i][i] is '-', which is not 'o', so it's correctly ignored.
    wins[i] = S[i].count('o')

# Create a list of tuples to store ranking information
# Each tuple will be (negative_wins, player_id)
# We use negative wins so that sorting the list in ascending order
# will place players with more wins earlier (since -5 < -4).
# The player_id is included as the second element. In case of a tie in wins (i.e., negative_wins are equal),
# the default tuple sorting will sort by the second element (player_id) in ascending order,
# which matches the tie-breaking rule (smaller player number ranks higher).
player_info = []
for i in range(N):
    # Player IDs are 1-based, so player corresponding to index i is i+1
    player_info.append((-wins[i], i + 1))

# Sort the list of tuples
# Sorting is done first by the number of wins (descending, due to negative sign)
# and then by player ID (ascending) in case of ties.
player_info.sort()

# Extract the player IDs from the sorted list
# The player IDs are the second element in each tuple
sorted_player_ids = [player_id for (neg_wins, player_id) in player_info]

# Print the sorted player IDs separated by spaces
# The * operator unpacks the list elements as separate arguments to the print function
print(*sorted_player_ids)
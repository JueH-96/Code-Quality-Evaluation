import sys

# Read the input line from standard input
line = sys.stdin.readline().split()

# Convert the parts into integers
N = int(line[0])
T = int(line[1])
A = int(line[2])

# In a mayoral election with N total votes, where N is odd, a candidate wins
# if they receive strictly more than N/2 votes.
# Using integer arithmetic, this means a candidate needs V votes such that 2 * V > N.
# Since N is odd, the minimum number of votes required to win is (N // 2) + 1.
# For example, if N=7, N//2 = 3, so 3 + 1 = 4 votes are needed. 2*4 > 7 is true.
# If N=1, N//2 = 0, so 0 + 1 = 1 vote is needed. 2*1 > 1 is true.
# The condition 2 * V > N correctly identifies if V is a winning number of votes.

# The outcome of the election is already decided if one candidate has
# accumulated enough votes that they are guaranteed to win, regardless
# of how the remaining votes are distributed.
# This happens if a candidate's current vote count is already greater than N/2.

# Check if Takahashi's current votes (T) already guarantee a win.
# This happens if 2 * T > N.
takahashi_wins_for_sure = (2 * T > N)

# Check if Aoki's current votes (A) already guarantee a win.
# This happens if 2 * A > N.
aoki_wins_for_sure = (2 * A > N)

# The outcome is decided if either Takahashi or Aoki has already secured enough votes to win.
if takahashi_wins_for_sure or aoki_wins_for_sure:
    print("Yes")
else:
    print("No")
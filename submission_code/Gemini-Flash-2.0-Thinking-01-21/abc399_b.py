# YOUR CODE HERE
import sys
from collections import Counter

# Read N, the number of people participating in the contest.
# People are labeled from 1 to N.
N = int(sys.stdin.readline())

# Read the scores P_1, P_2, ..., P_N.
# The input line contains N space-separated integers.
# We store these scores in a list. The i-th element in the list (0-indexed)
# corresponds to the score of the person who is originally person i+1.
scores = list(map(int, sys.stdin.readline().split()))

# Calculate the frequency of each score.
# This helps in grouping people by score, which is essential for the ranking procedure.
# collections.Counter is a convenient way to do this.
score_counts = Counter(scores)

# Get the unique score values and sort them in descending order.
# The ranking procedure processes people with the maximum score first.
# By iterating through scores from highest to lowest, we ensure that we always
# find the maximum score among the currently unranked people.
unique_scores = sorted(score_counts.keys(), reverse=True)

# Calculate the rank assigned to each unique score value.
# This dictionary will map a specific score to the rank it receives.
score_to_rank = {}
current_rank = 1 # Initialize the rank for the highest scoring group to 1.

# Iterate through the unique scores in descending order.
for score in unique_scores:
    # All people who achieved this 'score' are assigned the 'current_rank'.
    score_to_rank[score] = current_rank

    # Get the number of people who have this score.
    # This is the size of the group receiving the 'current_rank'.
    count = score_counts[score]

    # According to the procedure, the rank for the *next* group of people
    # (those with the next lower score) is incremented by the number of people
    # who were just ranked.
    current_rank += count

# Determine the final rank for each person based on their original score.
# The problem requires printing the rank of person i on the i-th line (1-indexed).
# Since the input list 'scores' maintains the original order of persons (score of person 1,
# followed by score of person 2, and so on), we can iterate through this list.
# Looking up the rank for each score value will produce the ranks in the correct
# output order corresponding to persons 1 through N.
person_ranks = []
for score in scores:
    # For the current person (whose score is 'score' from the original list),
    # look up the rank assigned to this score value using the mapping calculated earlier.
    rank = score_to_rank[score]
    person_ranks.append(rank)

# Print the final ranks, one per line.
# The list person_ranks is already ordered correctly to match the original
# person indices (1st element is rank of person 1, 2nd is rank of person 2, etc.).
for rank in person_ranks:
    # Use sys.stdout.write for potentially faster output in competitive programming.
    # Add '
' to ensure each rank is on a new line.
    sys.stdout.write(str(rank) + '
')
# Read N and X from the first line of input
# N is the number of problems, X is the score threshold
N, X = map(int, input().split())

# Read the list of scores S from the second line of input
# S_i in the problem description corresponds to an element in this list S
S = list(map(int, input().split()))

# Initialize the total score for problems meeting the criteria
total_score_matching_criteria = 0

# Iterate through each score in the list S
for score in S:
    # Check if the current score is less than or equal to the threshold X
    if score <= X:
        # If it is, add it to the total_score_matching_criteria
        total_score_matching_criteria += score

# Print the final calculated total score
print(total_score_matching_criteria)
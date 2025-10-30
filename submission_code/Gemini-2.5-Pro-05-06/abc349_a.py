# Read the number of people, N
N = int(input())

# Read the N-1 scores A_1, A_2, ..., A_{N-1}
# These are provided on a single line, separated by spaces.
# input().split() creates a list of strings.
# map(int, ...) converts each string in the list to an integer.
# list(...) converts the map object to a list of integers.
A_values = list(map(int, input().split()))

# Calculate the sum of the known scores
sum_of_known_A_values = sum(A_values)

# The score of person N is the negative of this sum,
# because the total sum of scores for all N people must be 0.
score_of_person_N = -sum_of_known_A_values

# Print the calculated score for person N
print(score_of_person_N)
# Read the input values
N, L = map(int, input().split())
scores = list(map(int, input().split()))

# Count the number of people who passed the exam
pass_count = sum(score >= L for score in scores)

# Print the result
print(pass_count)
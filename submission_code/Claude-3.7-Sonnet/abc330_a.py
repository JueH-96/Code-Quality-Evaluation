# Read the first line containing N and L
N, L = map(int, input().split())

# Read the second line containing the scores of all N people
scores = list(map(int, input().split()))

# Count how many people scored at least L points
passed_count = sum(1 for score in scores if score >= L)

# Print the answer
print(passed_count)
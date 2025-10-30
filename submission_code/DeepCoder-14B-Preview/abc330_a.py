# Read the number of people and the minimum passing score
n, l = map(int, input().split())

# Read the list of scores
scores = list(map(int, input().split()))

# Count how many scores are at least L
count = 0
for score in scores:
    if score >= l:
        count += 1

# Print the result
print(count)
# Read the input values
n, l = map(int, input().split())
scores = list(map(int, input().split()))

# Count the number of scores meeting or exceeding the passing threshold
count = 0
for score in scores:
    if score >= l:
        count += 1

# Output the result
print(count)
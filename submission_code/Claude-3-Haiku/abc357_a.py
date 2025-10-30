# Read the input
n, m = map(int, input().split())
h = list(map(int, input().split()))

# Initialize the count of aliens who can disinfect all their hands
count = 0

# Iterate through the aliens
for hands in h:
    if m >= hands:
        m -= hands
        count += 1
    else:
        break

# Print the result
print(count)
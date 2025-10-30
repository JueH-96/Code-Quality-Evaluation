n = int(input())
a = list(map(int, input().split()))

# Track positions for each number
positions = {}
for idx, num in enumerate(a):
    if num not in positions:
        positions[num] = []
    positions[num].append(idx + 1)  # Convert to 1-based index

# Collect the middle occurrence for each number and prepare for sorting
sorted_numbers = []
for i in range(1, n + 1):
    mid = positions[i][1]
    sorted_numbers.append((mid, i))

# Sort by the middle index, then by the number if needed
sorted_numbers.sort()

# Extract the numbers in the required order
result = [str(num) for (_, num) in sorted_numbers]

print(' '.join(result))
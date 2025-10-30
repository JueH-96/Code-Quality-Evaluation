# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Find all positions for each number
positions = {}
for i in range(len(a)):
    num = a[i]
    if num not in positions:
        positions[num] = []
    positions[num].append(i + 1)  # Convert to 1-indexed

# Calculate f(i) for each number i
f_values = {}
for i in range(1, n + 1):
    pos_list = positions[i]
    pos_list.sort()  # Should already be sorted, but just to be safe
    f_values[i] = pos_list[1]  # Middle occurrence (second element)

# Sort numbers 1 to N by their f(i) values
numbers = list(range(1, n + 1))
numbers.sort(key=lambda x: f_values[x])

# Output the result
print(' '.join(map(str, numbers)))
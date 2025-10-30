n = int(input())
a = list(map(int, input().split()))

# Find occurrences of each number
occurrences = [[] for _ in range(n + 1)]
for i in range(len(a)):
    occurrences[a[i]].append(i + 1)  # Store 1-based indices

# Calculate f(i) for each number
f_values = [0] * (n + 1)
for i in range(1, n + 1):
    f_values[i] = occurrences[i][1]  # Middle occurrence (index 1 in 0-based list)

# Sort numbers by their f values
numbers = list(range(1, n + 1))
numbers.sort(key=lambda x: f_values[x])

# Output
print(*numbers)
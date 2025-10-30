# Below is your solution to the problem specified above

N = int(input())
data = []
for _ in range(N):
    S, A = input().split()
    A = int(A)
    data.append((A, S))

# Find the youngest person's index
youngest_index = min(range(N), key=lambda i: data[i])

# Sort by age, then by the original order (clockwise)
sorted_data = sorted(data, key=lambda x: (x[0], (youngest_index - data.index(x) + N) % N))

# Print the names in sorted order, in terms of seating positions
for _, name in sorted_data:
    print(name)
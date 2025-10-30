# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
c = list(map(int, input().split()))

# Find positions that need to be flipped
flip_1_to_0 = []  # positions where a[i] = 1 and b[i] = 0
flip_0_to_1 = []  # positions where a[i] = 0 and b[i] = 1

for i in range(n):
    if a[i] != b[i]:
        if a[i] == 1:  # need to flip from 1 to 0
            flip_1_to_0.append(i)
        else:  # need to flip from 0 to 1
            flip_0_to_1.append(i)

# Sort by cost
# For 1->0 flips, sort by decreasing cost (flip high cost first)
flip_1_to_0.sort(key=lambda i: c[i], reverse=True)
# For 0->1 flips, sort by increasing cost (flip low cost last)
flip_0_to_1.sort(key=lambda i: c[i])

# Combine the flip order
flip_order = flip_1_to_0 + flip_0_to_1

# Calculate total cost
total_cost = 0
current_a = a.copy()

for i in flip_order:
    # Flip position i
    current_a[i] = 1 - current_a[i]
    
    # Calculate cost for this operation
    operation_cost = sum(current_a[j] * c[j] for j in range(n))
    total_cost += operation_cost

print(total_cost)
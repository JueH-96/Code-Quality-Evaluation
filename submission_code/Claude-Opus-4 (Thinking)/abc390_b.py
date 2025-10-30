# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

is_geometric = True

# Check if all consecutive ratios are equal
# Using cross multiplication: A_{i+1} / A_i = A_2 / A_1
# Which means A_{i+1} * A_1 = A_i * A_2
for i in range(1, n - 1):
    if a[i + 1] * a[0] != a[i] * a[1]:
        is_geometric = False
        break

print("Yes" if is_geometric else "No")
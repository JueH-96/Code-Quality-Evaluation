# YOUR CODE HERE
n, m = map(int, input().split())
x = list(map(int, input().split()))

# Calculate the initial length of the tour
length = 0
for i in range(m-1):
    length += abs(x[i] - x[i+1])

# Calculate the minimum possible length of the tour by considering each bridge closure
min_length = length
for i in range(m):
    a, b = x[i], x[(i+1) % m]
    # Calculate the length of the tour if the bridge between islands a and b is closed
    length = length - abs(a - b) + 2 * min(abs(a - 1), abs(b - 1), abs(a - n), abs(b - n))
    min_length = min(min_length, length)

print(min_length)
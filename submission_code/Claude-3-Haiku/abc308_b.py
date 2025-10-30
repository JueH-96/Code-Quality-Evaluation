# Read input
N, M = map(int, input().split())
C = input().split()
D = input().split()
P = list(map(int, input().split()))

# Initialize total price
total_price = 0

# Iterate through the plates
for c in C:
    # Check if the color matches any of the special colors
    if c in D:
        index = D.index(c)
        total_price += P[index + 1]
    else:
        total_price += P[0]

# Print the total price
print(total_price)
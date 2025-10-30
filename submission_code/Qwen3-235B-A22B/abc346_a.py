# Read input values
n = int(input())
a = list(map(int, input().split()))

# Calculate the products of consecutive elements
products = []
for i in range(n - 1):
    products.append(str(a[i] * a[i + 1]))

# Print the products separated by spaces
print(' '.join(products))
# Read the input
n = int(input())
a = list(map(int, input().split()))

# Calculate the B_i values
b = [a[i] * a[i+1] for i in range(n-1)]

# Print the result
print(" ".join(map(str, b)))
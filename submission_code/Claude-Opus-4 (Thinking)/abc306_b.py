# YOUR CODE HERE
# Read the sequence of 64 binary digits
A = list(map(int, input().split()))

# Calculate the sum A_0*2^0 + A_1*2^1 + ... + A_63*2^63
result = 0
for i in range(64):
    if A[i] == 1:
        result += 2**i

# Print the result
print(result)
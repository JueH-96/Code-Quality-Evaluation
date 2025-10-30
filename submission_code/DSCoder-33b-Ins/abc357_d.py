# YOUR CODE HERE

N = int(input())

# Calculate V_N
V_N = int(str(N) * N)

# Calculate the remainder when V_N is divided by 998244353
remainder = V_N % 998244353

print(remainder)
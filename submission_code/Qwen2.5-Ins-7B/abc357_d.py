# YOUR CODE HERE
N = int(input())
mod = 998244353

# Calculate the remainder of V_N divided by 998244353
remainder = pow(10, N, mod) - 1
result = remainder * pow(10 - 1, mod - 2, mod) % mod

print(result)
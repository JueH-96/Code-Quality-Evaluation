# Read the input from stdin
N = int(input())

# Calculate the remainder of V_N when divided by 998244353
mod = 998244353
# Convert N to a string and repeat it N times
V_N_str = str(N) * N
# Convert the repeated string back to an integer
V_N = int(V_N_str)
# Calculate the remainder
remainder = V_N % mod

# Print the remainder to stdout
print(remainder)
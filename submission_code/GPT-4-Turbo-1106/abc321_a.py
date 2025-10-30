# Read input from stdin
N = input()

# Check if N is a 321-like Number
is_321_like = all(N[i] > N[i+1] for i in range(len(N) - 1))

# Print Yes if it is a 321-like Number, No otherwise
print("Yes" if is_321_like else "No")
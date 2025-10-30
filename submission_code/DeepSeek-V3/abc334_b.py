# YOUR CODE HERE
A, M, L, R = map(int, input().split())

# Calculate the first tree position >= L
# The general form is A + k*M >= L
# Solving for k: k >= (L - A) / M
# Since k must be an integer, we take the ceiling of (L - A) / M
# But since we are dealing with integers, we can use the formula:
# k_min = (L - A + M - 1) // M if M > 0 else (L - A) // M
# However, since M is always positive (1 <= M <= 1e9), we can simplify:
k_min = (L - A + M - 1) // M

# Calculate the first tree position <= R
# The general form is A + k*M <= R
# Solving for k: k <= (R - A) / M
# Since k must be an integer, we take the floor of (R - A) / M
k_max = (R - A) // M

# The number of trees is k_max - k_min + 1
if k_min > k_max:
    print(0)
else:
    print(k_max - k_min + 1)
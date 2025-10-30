# YOUR CODE HERE
A, M, L, R = map(int, input().split())

# Calculate the first tree position >= L
if L <= A:
    k_min = (L - A) // M
    if A + k_min * M < L:
        k_min += 1
else:
    k_min = (L - A + M - 1) // M

# Calculate the last tree position <= R
if R >= A:
    k_max = (R - A) // M
else:
    k_max = (R - A - M + 1) // M

# Ensure that k_min <= k_max
if k_min > k_max:
    print(0)
else:
    print(k_max - k_min + 1)
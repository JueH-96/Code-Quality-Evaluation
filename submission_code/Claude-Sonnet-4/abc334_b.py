A, M, L, R = map(int, input().split())

# Find the range of k values
# We need: L <= A + k*M <= R
# Which gives us: (L-A)/M <= k <= (R-A)/M

# Find minimum k (ceiling of (L-A)/M)
if (L - A) % M == 0:
    k_min = (L - A) // M
else:
    if L - A > 0:
        k_min = (L - A) // M + 1
    else:
        k_min = (L - A) // M

# Find maximum k (floor of (R-A)/M)
k_max = (R - A) // M

# Count of valid k values
if k_min <= k_max:
    result = k_max - k_min + 1
else:
    result = 0

print(result)
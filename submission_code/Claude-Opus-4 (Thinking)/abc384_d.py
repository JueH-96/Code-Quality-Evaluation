# YOUR CODE HERE
n, s = map(int, input().split())
a = list(map(int, input().split()))

# Compute sum of one full period
sum_period = sum(a)

# Extend array for easier computation
extended = a + a

# Check all possible subsequences
found = False

# For each starting position in the period
for start in range(n):
    # Check lengths 1 to n
    current_sum = 0
    for length in range(1, n + 1):
        current_sum += extended[start + length - 1]
        if current_sum == s:
            found = True
            break
    if found:
        break
    
    # Check longer lengths (k >= 1)
    if sum_period != 0:
        # For each possible remainder r (0 to n-1)
        remainder_sum = 0
        for r in range(n):
            if r > 0:
                remainder_sum += extended[start + r - 1]
            # Length = k*n + r, where k >= 1
            # We need k * sum_period + remainder_sum = s
            # So k = (s - remainder_sum) / sum_period
            if (s - remainder_sum) % sum_period == 0:
                k = (s - remainder_sum) // sum_period
                if k > 0:  # Ensure k >= 1
                    found = True
                    break
        if found:
            break

print("Yes" if found else "No")
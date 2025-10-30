N, T, P = map(int, input().split())
hair_lengths = list(map(int, input().split()))

# Calculate days needed for each person to reach hair length T
days_needed = [max(0, T - length) for length in hair_lengths]

# Sort days in ascending order
days_needed.sort()

# The answer is the day when the P-th person reaches length T
print(days_needed[P - 1])
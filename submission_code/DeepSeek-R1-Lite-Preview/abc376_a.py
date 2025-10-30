# Read input values
N_C = input().split()
N = int(N_C[0])
C = int(N_C[1])

T = list(map(int, input().split()))

# Initialize variables
last_candy_time = -1
candies = 0

# Process each press time
for t in T:
    if last_candy_time == -1:
        candies += 1
        last_candy_time = t
    elif t - last_candy_time >= C:
        candies += 1
        last_candy_time = t

# Output the result
print(candies)
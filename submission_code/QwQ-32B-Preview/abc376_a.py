# Read N and C from input
N_C_input = input().split()
N = int(N_C_input[0])
C = int(N_C_input[1])

# Read the list of T values
T = list(map(int, input().split()))

# Initialize variables
last_candy_time = T[0]
candy_count = 1  # First candy is always received

# Iterate through the presses starting from the second one
for i in range(1, N):
    time_since_last = T[i] - last_candy_time
    if time_since_last >= C:
        candy_count += 1
        last_candy_time = T[i]

# Output the result
print(candy_count)
# Read input
N_C = input().split()
N = int(N_C[0])
C = int(N_C[1])
T = list(map(int, input().split()))

# Initialize variables
count = 1
last_time = T[0]

# Iterate through the presses starting from the second one
for i in range(1, N):
    current_time = T[i]
    if current_time - last_time >= C:
        count += 1
        last_time = current_time

# Output the result
print(count)
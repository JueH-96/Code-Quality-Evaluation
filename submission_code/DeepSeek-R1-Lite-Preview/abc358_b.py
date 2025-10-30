# Read the first line and split into N and A
N_A = input().split()
N = int(N_A[0])
A = int(N_A[1])

# Read the second line and split into list T
T = list(map(int, input().split()))

# Initialize the current finish time
current_finish_time = 0

# Iterate through each person's arrival time
for Ti in T:
    if Ti >= current_finish_time:
        finish_time = Ti + A
    else:
        finish_time = current_finish_time + A
    print(finish_time)
    current_finish_time = finish_time
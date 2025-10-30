# Read input values
N, M = map(int, input().split())
firework_days = list(map(int, input().split()))

# Initialize a list to store the results
results = [0] * N

# Initialize a pointer for the firework_days list
firework_pointer = 0

# Loop through each day
for i in range(N):
    # If the current day is a firework day, print 0 and move the pointer
    if firework_pointer < M and firework_days[firework_pointer] == i + 1:
        print(0)
        firework_pointer += 1
    else:
        # Otherwise, find the next firework day and print the difference
        print(firework_days[firework_pointer] - (i + 1))
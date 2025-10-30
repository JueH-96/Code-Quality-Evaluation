# Read the input
N, Q = map(int, input().split())
T = list(map(int, input().split()))

# Initialize the teeth count
teeth = [1] * N

# Perform the treatments
for t in T:
    if teeth[t-1] == 1:
        teeth[t-1] = 0
    else:
        teeth[t-1] = 1

# Count the remaining teeth
print(sum(teeth))
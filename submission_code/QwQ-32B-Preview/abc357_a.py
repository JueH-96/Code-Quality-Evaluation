# Read the first line and split into N and M
N_M = input().split()
N = int(N_M[0])
M = int(N_M[1])

# Read the second line and split into list of H_i
H = list(map(int, input().split()))

# Initialize remaining disinfectant
remaining_disinfectant = M

# Initialize count of aliens who can disinfect all their hands
count = 0

# Iterate through each alien's hands
for h in H:
    if remaining_disinfectant >= h:
        remaining_disinfectant -= h
        count += 1
    else:
        remaining_disinfectant = 0

# Print the result
print(count)
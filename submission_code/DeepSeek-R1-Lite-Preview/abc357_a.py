# Read the first line containing N and M
N, M = map(int, input().split())

# Read the second line containing H_i values
H = list(map(int, input().split()))

# Initialize counter for successful aliens
counter = 0

# Iterate through each alien's hands
for h in H:
    if M >= h:
        M -= h
        counter += 1
    else:
        M = 0  # Alien uses up the remaining disinfectant

# Print the number of successful aliens
print(counter)
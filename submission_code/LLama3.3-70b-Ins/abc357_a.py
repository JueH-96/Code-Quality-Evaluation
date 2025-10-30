# Read the number of aliens and the total disinfectant capacity
N, M = map(int, input().split())

# Read the number of hands for each alien
hands = list(map(int, input().split()))

# Initialize the count of aliens who can disinfect all their hands
count = 0

# Iterate over the aliens
for hand in hands:
    # If the disinfectant is enough to disinfect all hands of the current alien
    if M >= hand:
        # Disinfect the alien's hands
        M -= hand
        # Increment the count of aliens who can disinfect all their hands
        count += 1
    else:
        # If the disinfectant is not enough, break the loop
        break

# Print the count of aliens who can disinfect all their hands
print(count)
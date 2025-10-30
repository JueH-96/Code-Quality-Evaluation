# Read N and M from the first line of input
N, M = map(int, input().split())

# Read the list of H values (hands for each alien) from the second line of input
H = list(map(int, input().split()))

# Initialize the amount of disinfectant remaining
disinfectant_remaining = M

# Initialize a counter for aliens who successfully disinfected all their hands
successful_aliens_count = 0

# Iterate through each alien's hand requirements
for hands_needed in H:
    # Check if there is enough disinfectant remaining for this alien
    # to disinfect all their hands.
    if disinfectant_remaining >= hands_needed:
        # If yes, this alien successfully disinfects all their hands.
        successful_aliens_count += 1
    
    # Regardless of whether the alien succeeded in disinfecting all their hands,
    # they use up the disinfectant equivalent to the number of hands they have.
    # The problem states "they will use up the remaining disinfectant" even if not enough,
    # and the sample explanation clarifies that the bottle capacity effectively decreases
    # by H_i, possibly going into negative numbers, implying the bottle is empty
    # and more was needed.
    disinfectant_remaining -= hands_needed

# Print the total number of aliens who successfully disinfected all their hands
print(successful_aliens_count)
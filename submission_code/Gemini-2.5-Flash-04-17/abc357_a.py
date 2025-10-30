import sys

# Read N (number of aliens) and M (disinfectant capacity)
line1 = sys.stdin.readline().split()
N = int(line1[0])
M = int(line1[1])

# Read the list of hand counts for each alien
line2 = sys.stdin.readline().split()
H = [int(h) for h in line2]

# Initialize counter for the number of aliens who can disinfect all their hands
successful_aliens_count = 0

# Keep track of the remaining disinfectant
remaining_disinfectant = M

# Process each alien in order
for hands_needed in H:
    # Check if there is enough disinfectant for the current alien to disinfect all their hands
    if remaining_disinfectant >= hands_needed:
        # Yes, enough disinfectant. The alien disinfects all hands successfully.
        remaining_disinfectant -= hands_needed
        successful_aliens_count += 1
    else:
        # No, not enough disinfectant for this alien to disinfect all their hands fully.
        # According to the problem, they still use up the remaining disinfectant.
        # This means remaining_disinfectant becomes 0.
        # Once remaining_disinfectant is 0, no subsequent alien can possibly disinfect any hands (since H_i >= 1),
        # let alone all of them. So, we can stop counting successful aliens here.
        # remaining_disinfectant = 0 # This line is implicitly handled by the break below
        break # Stop processing for successful alien count as no more successes are possible

# Print the final count of successful aliens
print(successful_aliens_count)
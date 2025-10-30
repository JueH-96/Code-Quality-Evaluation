# YOUR CODE HERE
def solve():
    # Read N and M from the first line
    N, M = map(int, input().split())
    
    # Read the list of hand counts H_i from the second line
    H_list = list(map(int, input().split()))
    
    disinfected_aliens_count = 0
    remaining_disinfectant = M
    
    # Iterate through each alien
    for hands_of_current_alien in H_list:
        # Check if there is enough disinfectant for the current alien
        if remaining_disinfectant >= hands_of_current_alien:
            # If yes, this alien disinfects all their hands
            remaining_disinfectant -= hands_of_current_alien
            disinfected_aliens_count += 1
        else:
            # If no, this alien cannot disinfect all their hands.
            # They will use up the remaining disinfectant (making it 0).
            # No subsequent alien will be able to disinfect all their hands.
            # So, we can stop processing.
            break
            
    # Print the total number of aliens who disinfected all their hands
    print(disinfected_aliens_count)

# Call the solve function to execute the logic
solve()
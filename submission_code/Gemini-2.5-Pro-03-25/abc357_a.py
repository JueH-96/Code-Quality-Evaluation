# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, simulates the disinfection process, and prints the result.
    """
    # Read N and M from the first line
    line1 = sys.stdin.readline().split()
    n = int(line1[0])  # Number of aliens
    m = int(line1[1])  # Initial amount of disinfectant

    # Read the list of hand counts H from the second line
    line2 = sys.stdin.readline().split()
    h = [int(x) for x in line2] # List of hand counts for each alien

    # Initialize the count of successfully disinfected aliens
    successful_aliens = 0
    # Initialize the remaining disinfectant
    remaining_disinfectant = m

    # Iterate through the aliens one by one
    for hands_needed in h:
        # Check if there is enough disinfectant for the current alien's hands
        if remaining_disinfectant >= hands_needed:
            # If yes, the alien disinfects all hands successfully
            remaining_disinfectant -= hands_needed # Decrease disinfectant amount
            successful_aliens += 1                 # Increment the count
        else:
            # If no, this alien cannot disinfect all hands.
            # Since aliens come one by one, no subsequent alien can succeed either.
            # Stop the process.
            break

    # Print the total number of aliens who successfully disinfected all their hands
    print(successful_aliens)

# Execute the solve function when the script runs
if __name__ == "__main__":
    solve()
# YOUR CODE HERE
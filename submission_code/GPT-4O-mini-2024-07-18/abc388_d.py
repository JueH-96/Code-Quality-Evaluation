def calculate_stones_after_years(N, A):
    # Initialize the result array with the initial stones
    B = A[:]
    
    # We will keep track of the total stones that will be given away
    total_stones_given = 0
    
    # Iterate through each alien
    for i in range(N):
        # The current alien will become an adult after (i + 1) years
        # They will receive stones from all previous adults
        if i > 0:  # Only if there are previous aliens
            total_stones_given += min(B[i - 1], 1)  # Only one stone can be given by each adult
        
        # Update the stones for the current alien
        B[i] += total_stones_given
        
        # After they become an adult, they will give away one stone if they have any
        if B[i] > 0:
            B[i] -= 1
    
    return B

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))

# Calculate the result
result = calculate_stones_after_years(N, A)

# Print the result
print(" ".join(map(str, result)))
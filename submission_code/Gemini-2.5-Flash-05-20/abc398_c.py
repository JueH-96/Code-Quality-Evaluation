import sys
from collections import Counter

def solve():
    # Read N
    N = int(sys.stdin.readline())

    # Read the list of integers A
    A = list(map(int, sys.stdin.readline().split()))

    # Step 1: Count the frequencies of all numbers in list A
    # This helps in quickly determining if a number appears exactly once.
    counts = Counter(A)

    # Initialize variables to keep track of the person with the greatest unique integer
    max_unique_val = -1      # Stores the maximum integer value found so far that is unique
    result_label = -1        # Stores the 1-indexed label of the person corresponding to max_unique_val

    # Step 2: Iterate through each person (and their value) to find the best candidate
    # Labels are 1-indexed, so A[i] corresponds to person (i+1)
    for i in range(N):
        current_val = A[i]          # The integer value of the current person
        current_label = i + 1       # The 1-indexed label of the current person

        # Check if the current value appears exactly once in the entire list A
        if counts[current_val] == 1:
            # If it is unique, check if its value is greater than the current maximum unique value found
            if current_val > max_unique_val:
                max_unique_val = current_val  # Update the maximum unique value
                result_label = current_label  # Update the label of the person with this value
    
    # Step 3: Print the result
    # If result_label is still -1, it means no person satisfied the unique condition.
    sys.stdout.write(str(result_label) + "
")

# Call the solve function to run the program
solve()
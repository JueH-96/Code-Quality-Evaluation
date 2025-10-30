# YOUR CODE HERE
import sys

def solve():
    # Read N and K from the first line
    line1 = sys.stdin.readline().split()
    n = int(line1[0])
    k = int(line1[1])

    # Read the string S from the second line and convert to a list of characters
    # Use strip() to remove potential trailing newline characters
    s = list(sys.stdin.readline().strip())

    count = 0 # Initialize the total count of strawberries eaten

    # Outer loop: Continue as long as we can find and process a group of K healthy teeth
    while True:
        # Flag to track if we found and ate a strawberry in this scan pass
        found_strawberry_in_pass = False
        
        # Inner loop (scan pass): Iterate through possible starting positions
        # for K consecutive teeth
        # A substring of length K starting at index i ends at i + K - 1.
        # The last possible starting index is when i + K - 1 is the last index of the string (n-1).
        # So, i + K - 1 = n - 1 => i = n - K. The loop goes up to n - K (inclusive).
        # If n < k, range(n - k + 1) will be range(a_negative_or_zero_number), which results in an empty range.
        # This is correct as no strawberry can be eaten if N < K.
        for i in range(n - k + 1):
            # Check if the substring of length K starting at index i consists of only 'O's
            is_k_healthy = True
            for j in range(k):
                # Check the character at index i + j
                if s[i + j] == 'X':
                    is_k_healthy = False
                    break # No need to check further if one tooth is 'X'

            # If K consecutive 'O's are found starting at index i
            if is_k_healthy:
                # We found an opportunity to eat a strawberry
                count += 1 # Increment the count
                found_strawberry_in_pass = True # Mark that we found one in this pass
                
                # Change the used K teeth from 'O' to 'X'
                for j in range(k):
                    s[i + j] = 'X'
                
                # We have modified the state of the teeth.
                # According to the greedy strategy, we take the first available opportunity.
                # After taking it, we need to rescan the entire string from the beginning
                # in the next pass of the outer loop to find the next available opportunity.
                # To do this, we break out of the current inner scan loop (the for i loop).
                break 

        # After completing an inner scan pass (either by finding a strawberry and breaking,
        # or by iterating through all possible starting positions without finding one),
        # check if any strawberry was eaten in this pass.
        if not found_strawberry_in_pass:
            # If no strawberry was found in the entire scan pass, it means
            # there are no more opportunities left. Exit the main loop.
            break 

    # Print the final maximum number of strawberries eaten
    print(count)

solve()
# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read the number of white keys (W) and black keys (B) from standard input
    w, b = map(int, sys.stdin.readline().split())
    
    # Define the repeating pattern of the infinite piano keyboard string S
    # The pattern is "wbwbwwbwbwbw"
    pattern = "wbwbwwbwbwbw"
    pattern_len = len(pattern) # The length of the pattern is 12
    
    # Calculate the total length of the substring we are looking for
    target_len = w + b
    
    # According to the problem constraints, W >= 0, B >= 0, and W + B >= 1.
    # This ensures that the target length (target_len) is at least 1.
    # If target_len were 0 (meaning W=0 and B=0), the empty string "" would trivially satisfy the condition.
    # However, this case is excluded by the constraint W + B >= 1.

    # We need to determine if there exists a substring of the infinite string S 
    # that has exactly W 'w's and B 'b's.
    # Since S is formed by repeating the pattern, any substring can be found within a sufficiently long prefix of S.
    # The maximum possible length of the target substring is W + B <= 100 + 100 = 200.
    
    # To guarantee that we can find any substring of length L (up to 200) starting at any possible position 
    # relative to the pattern's cycle (12 positions), we need a prefix of S that is long enough.
    # A prefix of length `pattern_len + max_target_len - 1` is sufficient.
    # Minimum required prefix length = 12 + 200 - 1 = 211 characters.
    
    # To be safe and simplify implementation, we can generate a prefix longer than this minimum.
    # Let's choose a prefix length of around 400 characters.
    
    # Calculate the number of times the pattern needs to be repeated to get a prefix of length >= 400.
    # num_repeats = ceil(400 / pattern_len) = ceil(400 / 12) = ceil(33.33...) = 34 repetitions.
    # Using 35 repetitions gives a prefix length of 12 * 35 = 420, which is amply sufficient.
    num_repeats = 35 
    s_prefix = pattern * num_repeats 

    # Initialize a boolean flag to track whether a matching substring has been found.
    found = False
    
    # Check if the target length is valid (it should be positive based on constraints).
    # Also, ensure the target length does not exceed the generated prefix length.
    # This check (target_len <= len(s_prefix)) is guaranteed to be true given the maximum W, B 
    # and our choice of prefix length.
    if target_len > 0 and target_len <= len(s_prefix):
        # Iterate through all possible starting indices `i` for a substring of length `target_len`
        # within the generated prefix `s_prefix`.
        # The loop should cover indices from 0 up to the last possible start index, 
        # which is `len(s_prefix) - target_len`.
        # The `range(N)` function generates integers from 0 to N-1. We need to go up to the index,
        # so the upper bound of the range should be `len(s_prefix) - target_len + 1`.
        for i in range(len(s_prefix) - target_len + 1):
            # Extract the substring of length `target_len` starting at index `i`.
            substring = s_prefix[i : i + target_len]
            
            # Count the number of 'w' characters in the extracted substring.
            # The `str.count()` method is efficient enough for the given constraints (L <= 200).
            current_w = substring.count('w')
            
            # Calculate the number of 'b' characters. Since the substring consists only of 'w' and 'b',
            # the count of 'b' is simply the total length minus the count of 'w'.
            # This avoids a second call to `count()`.
            current_b = target_len - current_w
            
            # Compare the counts (`current_w`, `current_b`) with the target values (W, B).
            if current_w == w and current_b == b:
                # If the counts match, we have found the required substring.
                # Set the flag to True and break out of the loop as no further search is needed.
                found = True
                break
            
    # After iterating through all possible substrings (or breaking early if found),
    # print the final result based on the value of the 'found' flag.
    if found:
        print("Yes")
    else:
        print("No")

# Execute the solve function to run the main logic of the program.
solve()
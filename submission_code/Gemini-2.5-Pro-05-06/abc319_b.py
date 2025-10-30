# Read the input integer N
N = int(input())

# List to store the characters of the resulting string
result_chars = []

# Iterate for each index i from 0 to N (inclusive)
for i in range(N + 1):
    # Default character for s_i is '-'
    char_for_i = '-'
    
    # Iterate through possible divisors j from 1 to 9
    # We want the smallest j, so we iterate in increasing order.
    for j in range(1, 10):  # j takes values 1, 2, ..., 9
        # Condition 1: j must be a divisor of N
        if N % j == 0:
            # If j is a divisor, calculate k = N/j.
            # N // j performs integer division. Since N % j == 0, it's exact.
            # N >= 1 and j >= 1, so k = N // j is always a positive integer.
            val_N_div_j = N // j
            
            # Condition 2: i must be a multiple of k (i.e., i % k == 0)
            # k is guaranteed to be non-zero (k >= 1), so i % k is safe.
            if i % val_N_div_j == 0:
                # If both conditions are met, this j is the smallest such j.
                char_for_i = str(j)  # Convert j to its string representation
                break  # Found the smallest j, so exit the inner loop for this i
                
    # Append the determined character for s_i to our list
    result_chars.append(char_for_i)

# Join all characters in the list to form the final string
output_string = "".join(result_chars)

# Print the final string
print(output_string)
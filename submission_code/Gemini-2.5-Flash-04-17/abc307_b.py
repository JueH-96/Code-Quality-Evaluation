# Read N
N = int(input())

# Read N strings into a list
strings = []
for _ in range(N):
    strings.append(input())

# Flag to indicate if a palindrome pair is found
found_palindrome = False

# Iterate through all possible first strings (index i)
for i in range(N):
    # Iterate through all possible second strings (index j)
    for j in range(N):
        # The problem requires distinct strings, corresponding to distinct indices
        if i == j:
            continue # Skip if the indices are the same
        
        # Concatenate the string at index i and the string at index j
        concat_str = strings[i] + strings[j]
        
        # Check if the concatenated string is a palindrome
        # A string is a palindrome if it is equal to its reverse
        if concat_str == concat_str[::-1]:
            # If a palindrome is found, set the flag
            found_palindrome = True
            # We only need to find one such pair, so we can stop searching
            break # Exit the inner loop
            
    # If a palindrome was found by the inner loop, exit the outer loop too
    if found_palindrome:
        break

# Print the final result based on the flag
if found_palindrome:
    print("Yes")
else:
    print("No")
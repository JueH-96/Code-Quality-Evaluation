# YOUR CODE HERE
# Read N
N = int(input())
# Read S
S = input()

# Flag to indicate if adjacent 'a' and 'b' are found
found_adjacent_ab = False

# Iterate through the string up to the second to last character
# The loop goes from index 0 to N-2, accessing S[i] and S[i+1]
for i in range(N - 1):
    # Get the current character and the next character
    char1 = S[i]
    char2 = S[i+1]

    # Check if the pair of adjacent characters is {'a', 'b'}
    # Using a set comparison automatically handles the order (both 'ab' and 'ba')
    if {char1, char2} == {'a', 'b'}:
        # Found an adjacent 'a' and 'b', set the flag and break the loop
        found_adjacent_ab = True
        break

# Print the final result based on the flag
if found_adjacent_ab:
    print("Yes")
else:
    print("No")
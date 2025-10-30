# Read N, the number of strings
N = int(input())

# Read the N strings into a list.
# S_list[k] will correspond to the (k+1)-th input string S_{k+1} from the problem statement.
S_list = []
for _ in range(N):
    S_list.append(input())

# Iterate through all possible pairs of distinct indices (idx1, idx2).
# These 0-based indices for S_list correspond to 1-based problem indices i=idx1+1 and j=idx2+1.
for idx1 in range(N):
    for idx2 in range(N):
        # The problem requires distinct integers i and j for S_i and S_j.
        # This means we need to pick strings from different positions in the input sequence.
        # So, the 0-based indices idx1 and idx2 must be different.
        if idx1 == idx2:
            continue

        # Get the strings S_i (which is S_list[idx1]) and S_j (which is S_list[idx2])
        string_i = S_list[idx1]
        string_j = S_list[idx2]

        # Concatenate them in the specified order: S_i followed by S_j
        concatenated_string = string_i + string_j

        # Check if the concatenated string is a palindrome.
        # A string t is a palindrome if it reads the same forwards and backwards.
        # In Python, this can be checked by comparing the string with its reverse (t[::-1]).
        if concatenated_string == concatenated_string[::-1]:
            # A pair satisfying the condition has been found.
            print("Yes")
            # Exit the program immediately, as we only need to determine existence.
            exit() # This will terminate the script.

# If the loops complete, it means no such pair (S_i, S_j) was found
# that forms a palindrome when concatenated.
print("No")
# YOUR CODE HERE
S = input()
T = input()

def is_subsequence(sub, main_string):
    """
    Checks if `sub` is a subsequence of `main_string`.
    `sub` and `main_string` are expected to be strings.
    """
    sub_ptr = 0
    main_ptr = 0
    
    while sub_ptr < len(sub) and main_ptr < len(main_string):
        if sub[sub_ptr] == main_string[main_ptr]:
            sub_ptr += 1
        main_ptr += 1
        
    return sub_ptr == len(sub)

# Condition 1: T is formed by taking a subsequence of length 3 from S
# and converting it to uppercase.
# This is equivalent to checking if T (converted to lowercase) is a subsequence of S.
# Example: S = "narita", T = "NRT". T.lower() = "nrt". Is "nrt" a subsequence of "narita"? Yes.
possible_by_cond1 = is_subsequence(T.lower(), S)

# Condition 2: T is formed by taking a subsequence of length 2 from S,
# converting it to uppercase, and appending 'X'.
# This means T's last character must be 'X', and T's first two characters (when lowercased)
# must form a subsequence of S.
# Example: S = "losangeles", T = "LAX".
# T[2] == 'X' is true.
# T[0:2].lower() = "la". Is "la" a subsequence of "losangeles"? Yes.
possible_by_cond2 = False
if T[2] == 'X':
    # Target for subsequence check is the first two characters of T, in lowercase.
    # T[0:2] extracts the first two characters (e.g., "LA" from "LAX").
    target_prefix_lower = T[0:2].lower()
    possible_by_cond2 = is_subsequence(target_prefix_lower, S)

if possible_by_cond1 or possible_by_cond2:
    print("Yes")
else:
    print("No")
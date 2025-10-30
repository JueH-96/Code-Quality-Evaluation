# YOUR CODE HERE
def are_chars_similar(char1, char2):
    # Condition 1: x and y are the same character.
    if char1 == char2:
        return True
    
    # Condition 2: One of x and y is '1' and the other is 'l'.
    if (char1 == '1' and char2 == 'l') or \
       (char1 == 'l' and char2 == '1'):
        return True
        
    # Condition 3: One of x and y is '0' and the other is 'o'.
    if (char1 == '0' and char2 == 'o') or \
       (char1 == 'o' and char2 == '0'):
        return True
        
    # If none of the conditions are met, they are not similar.
    return False

# Read N from input. N is the length of strings S and T.
N = int(input())

# Read strings S and T from input.
S = input()
T = input()

# Assume strings are similar until a non-similar pair of characters is found.
strings_are_similar_flag = True

# Iterate over the characters of S and T simultaneously.
# zip(S, T) will produce pairs (S[0], T[0]), (S[1], T[1]), ..., (S[N-1], T[N-1]).
# Since S and T are guaranteed to be of length N, this loop runs N times.
for s_char_i, t_char_i in zip(S, T):
    # Check if the current pair of characters is similar.
    if not are_chars_similar(s_char_i, t_char_i):
        # If any pair is not similar, the strings S and T are not similar.
        strings_are_similar_flag = False
        # No need to check further, so break out of the loop.
        break

# Print the final result based on the flag.
if strings_are_similar_flag:
    print("Yes")
else:
    print("No")
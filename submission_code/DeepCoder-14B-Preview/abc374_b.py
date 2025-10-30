# Read the input strings
S = input().strip()
T = input().strip()

# Iterate through each character index up to the maximum length of the two strings
for i in range(max(len(S), len(T))):
    # Get characters from S and T if they exist; else, use None
    s_char = S[i] if i < len(S) else None
    t_char = T[i] if i < len(T) else None
    
    # If the characters differ, print the 1-based index and exit
    if s_char != t_char:
        print(i + 1)
        exit()

# If all characters are the same and lengths are equal, print 0
print(0)
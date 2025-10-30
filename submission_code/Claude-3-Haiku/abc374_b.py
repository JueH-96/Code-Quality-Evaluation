# Read the input strings from stdin
S = input()
T = input()

# Check if the strings are equal
if S == T:
    print(0)
else:
    # Find the first character where they differ
    for i in range(min(len(S), len(T))):
        if S[i] != T[i]:
            print(i+1)
            break
    else:
        # If all characters are equal, print the length of the longer string
        if len(S) < len(T):
            print(len(T)+1)
        else:
            print(len(S)+1)
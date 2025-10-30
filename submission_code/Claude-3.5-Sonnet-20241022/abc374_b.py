# Read input strings
S = input().strip()
T = input().strip()

# If strings are equal, output 0
if S == T:
    print(0)
else:
    # Find the first position where strings differ
    min_len = min(len(S), len(T))
    
    # Check character by character up to the length of shorter string
    pos = 1
    while pos <= min_len:
        if S[pos-1] != T[pos-1]:
            print(pos)
            break
        pos += 1
    
    # If we haven't found a difference and reached here,
    # it means one string is longer than the other
    if pos > min_len:
        print(min_len + 1)
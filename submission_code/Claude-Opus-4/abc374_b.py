# YOUR CODE HERE
S = input().strip()
T = input().strip()

if S == T:
    print(0)
else:
    # Find the first position where they differ
    min_len = min(len(S), len(T))
    
    # Check character by character up to the shorter length
    for i in range(min_len):
        if S[i] != T[i]:
            print(i + 1)  # 1-indexed position
            break
    else:
        # If all characters match up to min_len, the difference is at position min_len + 1
        print(min_len + 1)
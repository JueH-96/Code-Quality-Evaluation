# YOUR CODE HERE
S = input().strip()
T = input().strip()

if S == T:
    print(0)
else:
    min_len = min(len(S), len(T))
    
    # Check character by character up to the shorter length
    for i in range(min_len):
        if S[i] != T[i]:
            print(i + 1)  # 1-indexed position
            break
    else:
        # All characters up to min_len are same, but lengths differ
        print(min_len + 1)
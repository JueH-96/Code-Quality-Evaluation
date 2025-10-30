# YOUR CODE HERE
S = input().strip()

# Check condition 1: length must be even
if len(S) % 2 != 0:
    print("No")
else:
    # Check condition 2: consecutive pairs must be equal
    valid = True
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            valid = False
            break
    
    if not valid:
        print("No")
    else:
        # Check condition 3: each character must appear exactly 0 or 2 times
        char_count = {}
        for char in S:
            char_count[char] = char_count.get(char, 0) + 1
        
        valid = True
        for count in char_count.values():
            if count != 2:
                valid = False
                break
        
        if valid:
            print("Yes")
        else:
            print("No")
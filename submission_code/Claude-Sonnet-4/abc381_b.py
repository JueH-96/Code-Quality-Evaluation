# YOUR CODE HERE
S = input().strip()

# Check condition 1: length must be even
if len(S) % 2 != 0:
    print("No")
else:
    # Check condition 2: adjacent pairs must be equal
    valid_pairs = True
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            valid_pairs = False
            break
    
    if not valid_pairs:
        print("No")
    else:
        # Check condition 3: each character appears exactly 0 or 2 times
        char_count = {}
        for char in S:
            char_count[char] = char_count.get(char, 0) + 1
        
        valid_counts = True
        for count in char_count.values():
            if count != 2:
                valid_counts = False
                break
        
        if valid_counts:
            print("Yes")
        else:
            print("No")
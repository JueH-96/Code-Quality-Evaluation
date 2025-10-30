S = input()

# Check condition 1: length is even
if len(S) % 2 != 0:
    print("No")
else:
    # Check condition 2: adjacent pairs are equal
    valid = True
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            valid = False
            break
    
    # Check condition 3: each character appears exactly 2 times
    if valid:
        char_count = {}
        for char in S:
            char_count[char] = char_count.get(char, 0) + 1
        
        for count in char_count.values():
            if count != 2:
                valid = False
                break
    
    print("Yes" if valid else "No")
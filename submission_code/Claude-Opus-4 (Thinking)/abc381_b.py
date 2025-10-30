# YOUR CODE HERE
s = input()

# Check if length is even
if len(s) % 2 != 0:
    print("No")
else:
    # Check if characters come in consecutive pairs
    valid_pairs = True
    for i in range(0, len(s), 2):
        if s[i] != s[i+1]:
            valid_pairs = False
            break
    
    if not valid_pairs:
        print("No")
    else:
        # Count character occurrences
        char_count = {}
        for char in s:
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
        
        # Check if all characters appear exactly 2 times
        valid_count = True
        for count in char_count.values():
            if count != 2:
                valid_count = False
                break
        
        if valid_count:
            print("Yes")
        else:
            print("No")
W, B = map(int, input().split())

pattern = "wbwbwwbwbwbw"

# Special cases where we need only one type of character
if B == 0:
    # Need only w's - check max consecutive w's in pattern
    max_consecutive_w = 0
    current_w = 0
    for char in pattern:
        if char == 'w':
            current_w += 1
            max_consecutive_w = max(max_consecutive_w, current_w)
        else:
            current_w = 0
    
    if W <= max_consecutive_w:
        print("Yes")
    else:
        print("No")
elif W == 0:
    # Need only b's - check max consecutive b's in pattern
    max_consecutive_b = 0
    current_b = 0
    for char in pattern:
        if char == 'b':
            current_b += 1
            max_consecutive_b = max(max_consecutive_b, current_b)
        else:
            current_b = 0
    
    if B <= max_consecutive_b:
        print("Yes")
    else:
        print("No")
else:
    # Need both w's and b's
    # Try all possible starting positions and lengths
    found = False
    
    # We need to check substrings of the repeated pattern
    # Since pattern length is 12, we check enough repetitions
    extended_pattern = pattern * (max(W + B, 20))
    
    for start in range(len(extended_pattern)):
        w_count = 0
        b_count = 0
        
        for end in range(start, min(start + W + B + 50, len(extended_pattern))):
            if extended_pattern[end] == 'w':
                w_count += 1
            else:
                b_count += 1
            
            if w_count == W and b_count == B:
                found = True
                break
            elif w_count > W or b_count > B:
                break
        
        if found:
            break
    
    if found:
        print("Yes")
    else:
        print("No")
t = int(input())
for _ in range(t):
    s = input().strip()
    
    # If already abc, no swap needed
    if s == "abc":
        print("YES")
        continue
        
    # Try all possible single swaps and check if any results in "abc"
    possible = False
    for i in range(3):
        for j in range(i+1, 3):
            # Create new string with characters at i and j swapped
            chars = list(s)
            chars[i], chars[j] = chars[j], chars[i]
            if ''.join(chars) == "abc":
                possible = True
                break
        if possible:
            break
            
    print("YES" if possible else "NO")
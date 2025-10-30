t = int(input())
for _ in range(t):
    s = input().strip()
    
    if s == "abc":
        print("YES")
        continue
    
    found = False
    
    # Try all possible swaps
    for i in range(3):
        for j in range(i+1, 3):
            # Create a new string with positions i and j swapped
            s_list = list(s)
            s_list[i], s_list[j] = s_list[j], s_list[i]
            if ''.join(s_list) == "abc":
                found = True
                break
        if found:
            break
    
    print("YES" if found else "NO")
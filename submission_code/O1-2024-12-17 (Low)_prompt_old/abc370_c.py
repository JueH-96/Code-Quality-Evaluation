def solve():
    import sys
    
    S = sys.stdin.readline().rstrip('
')
    T = sys.stdin.readline().rstrip('
')
    
    # If they are already equal, no changes are needed
    if S == T:
        print(0)
        return
    
    # Work with a list of characters for easy in-place modification
    s_list = list(S)
    X = []  # This will store the sequence of intermediate strings
    
    # Keep applying changes until s_list matches T
    while True:
        current_str = "".join(s_list)
        if current_str == T:
            break
        
        # Find the leftmost position at which T[i] < s_list[i]
        # among all differing positions
        idx = None
        for i in range(len(s_list)):
            if s_list[i] != T[i]:
                if T[i] < s_list[i]:
                    idx = i
                    break
        # If no such position was found, pick the leftmost difference
        if idx is None:
            for i in range(len(s_list)):
                if s_list[i] != T[i]:
                    idx = i
                    break
        
        # Make the change in s_list
        s_list[idx] = T[idx]
        X.append("".join(s_list))
    
    # Output results
    print(len(X))
    for x in X:
        print(x)
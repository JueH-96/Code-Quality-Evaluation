def solve():
    s = list(input())
    t = list(input())
    n = len(s)
    x = []
    
    while s != t:
        diff_indices = [i for i in range(n) if s[i] != t[i]]
        
        if not diff_indices:
            break
        
        best_s = list(s)
        best_s_str = "".join(best_s)
        
        
        for idx in diff_indices:
            temp_s = list(s)
            temp_s[idx] = t[idx]
            temp_s_str = "".join(temp_s)
            
            if temp_s_str < best_s_str:
                best_s = temp_s
                best_s_str = temp_s_str
        
        s = best_s
        x.append("".join(s))
        
    print(len(x))
    for item in x:
        print(item)

solve()
# YOUR CODE HERE
def solve():
    n = int(input())
    if n <= 9:
        print(n -1)
        return

    s = str(n)
    l = len(s)
    
    if l % 2 == 0:
        half_len = l // 2
        prefix = s[:half_len]
        
        ans1 = prefix + prefix[::-1]
        
        
        prefix_int = int(prefix)
        prefix_int += 1
        prefix_str = str(prefix_int)
        
        if len(prefix_str) > half_len:
            ans2 = "9" * (l-1)
        else:
            ans2 = prefix_str + prefix_str[::-1]
        
        
        prefix_int -= 2
        
        if prefix_int < 0:
            ans3 = "9" * (l-2) if l > 1 else "-1"
        else:
            prefix_str = str(prefix_int)
            
            if len(prefix_str) < half_len:
                prefix_str = "0" * (half_len - len(prefix_str)) + prefix_str
            ans3 = prefix_str + prefix_str[::-1]
        
        
        ans = min(ans1, ans2, ans3, key=lambda x: abs(int(x) - n) if x != "-1" else float('inf'))
        if ans == "-1":
            print("9" * (l-2))
        else:
            print(ans)
        
    else:
        half_len = l // 2
        prefix = s[:half_len]
        
        ans1 = prefix + s[half_len] + prefix[::-1]
        
        prefix_int = int(prefix + s[half_len])
        prefix_int += 1
        prefix_str = str(prefix_int)
        
        if len(prefix_str) > half_len + 1:
            ans2 = "9" * (l-1)
        else:
            ans2 = prefix_str[:-1] + prefix_str[::-1]
        
        prefix_int -= 2
        if prefix_int < 0:
            ans3 = "9" * (l-2) if l > 1 else "-1"
        else:
            prefix_str = str(prefix_int)
            if len(prefix_str) < half_len + 1:
                prefix_str = "0" * (half_len + 1 - len(prefix_str)) + prefix_str
            ans3 = prefix_str[:-1] + prefix_str[::-1]
        
        ans = min(ans1, ans2, ans3, key=lambda x: abs(int(x) - n) if x != "-1" else float('inf'))
        if ans == "-1":
            print("9" * (l-2))
        else:
            print(ans)

solve()
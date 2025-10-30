def solve():
    n, m = map(int, input().split())
    s = list(input())
    c = list(map(int, input().split()))
    
    indices = [[] for _ in range(m)]
    for i in range(n):
        indices[c[i]-1].append(i)
    
    for color_indices in indices:
        if not color_indices:
            continue
        
        last_char = s[color_indices[-1]]
        for i in range(len(color_indices) - 1, 0, -1):
            s[color_indices[i]] = s[color_indices[i-1]]
        s[color_indices[0]] = last_char
        
    print("".join(s))

solve()
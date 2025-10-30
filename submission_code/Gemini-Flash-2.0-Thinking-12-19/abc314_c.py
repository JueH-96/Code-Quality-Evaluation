def solve():
    n, m = map(int, input().split())
    s_str = input()
    colors = list(map(int, input().split()))
    s_list = list(s_str)
    
    for color_val in range(1, m + 1):
        indices = []
        for i in range(n):
            if colors[i] == color_val:
                indices.append(i)
        
        if not indices:
            continue
            
        color_chars = []
        for index in indices:
            color_chars.append(s_list[index])
            
        if len(color_chars) > 1:
            shifted_chars = [color_chars[-1]] + color_chars[:-1]
        else:
            shifted_chars = color_chars
            
        for i in range(len(indices)):
            s_list[indices[i]] = shifted_chars[i]
            
    print("".join(s_list))

if __name__ == '__main__':
    solve()
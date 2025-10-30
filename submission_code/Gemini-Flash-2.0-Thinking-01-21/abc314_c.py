def solve():
    n, m = map(int, input().split())
    s_str = input()
    colors = list(map(int, input().split()))
    s_list = list(s_str)
    
    color_indices = [[] for _ in range(m + 1)]
    for i in range(n):
        color_indices[colors[i]].append(i)
        
    for color_val in range(1, m + 1):
        indices_of_color = color_indices[color_val]
        if len(indices_of_color) >= 2:
            chars_of_color = []
            for index in indices_of_color:
                chars_of_color.append(s_list[index])
            
            shifted_chars = [chars_of_color[-1]] + chars_of_color[:-1]
            
            for i in range(len(indices_of_color)):
                s_list[indices_of_color[i]] = shifted_chars[i]
                
    print("".join(s_list))

if __name__ == '__main__':
    solve()
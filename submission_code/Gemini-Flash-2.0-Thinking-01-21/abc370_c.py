def solve():
    s_str = input()
    t_str = input()
    s_list = list(s_str)
    t_list = list(t_str)
    n = len(s_list)
    x_strings = []
    
    while s_list != t_list:
        diff_indices = []
        for i in range(n):
            if s_list[i] != t_list[i]:
                diff_indices.append(i)
        
        if not diff_indices:
            break
            
        next_strings = []
        for index in diff_indices:
            temp_s_list = list(s_list)
            temp_s_list[index] = t_list[index]
            next_strings.append("".join(temp_s_list))
            
        lex_smallest_next_s = min(next_strings)
        x_strings.append(lex_smallest_next_s)
        s_list = list(lex_smallest_next_s)
        
    print(len(x_strings))
    for s in x_strings:
        print(s)

if __name__ == '__main__':
    solve()
def solve():
    n = int(input())
    s_list = [input() for _ in range(n)]
    max_len = 0
    for s in s_list:
        max_len = max(max_len, len(s))
    m = max_len
    t_list = [""] * m
    for j in range(m):
        for k in range(1, n + 1):
            i = n - k + 1
            if 1 <= i <= n:
                if 1 <= j + 1 <= len(s_list[i-1]):
                    t_list[j] += s_list[i-1][j]
                elif len(s_list[i-1]) < j + 1 <= m:
                    t_list[j] += "*"
                    
    for j in range(m):
        current_t = list(t_list[j])
        while current_t and current_t[-1] == '*':
            current_t.pop()
        t_list[j] = "".join(current_t)
        
    output_t_list = [""] * m
    for j in range(m):
        output_t_list[j] = ""
        for k in range(1, n + 1):
            i = n - k + 1
            if 1 <= i <= n:
                if 1 <= j + 1 <= len(s_list[i-1]):
                    output_t_list[j] += s_list[i-1][j]
                elif len(s_list[i-1]) < j + 1 <= m:
                    output_t_list[j] += "*"

    result_t_list = []
    for t in output_t_list:
        temp_t = list(t)
        while temp_t and temp_t[-1] == '*':
            temp_t.pop()
        result_t_list.append("".join(temp_t))

    final_t_list = [""] * m
    for j in range(m):
        for k in range(1, n + 1):
            i = n - k + 1
            if 1 <= i <= n:
                if 1 <= j + 1 <= len(s_list[i-1]):
                    final_t_list[j] += s_list[i-1][j]
                elif len(s_list[i-1]) < j + 1 <= m:
                    final_t_list[j] += "*"

    trimmed_final_t_list = []
    for t in final_t_list:
        temp_t = list(t)
        while temp_t and temp_t[-1] == '*':
            temp_t.pop()
        trimmed_final_t_list.append("".join(temp_t))
        
    output_strings = []
    for j in range(m):
        t_str = ""
        for k in range(1, n + 1):
            i = n - k + 1
            char_to_add = '*'
            if 1 <= i <= n:
                if 1 <= j + 1 <= len(s_list[i-1]):
                    char_to_add = s_list[i-1][j]
                elif len(s_list[i-1]) < j + 1 <= m:
                    char_to_add = '*'
            t_str += char_to_add
        
        t_list_char = list(t_str)
        last_non_star_index = -1
        for index, char in enumerate(t_list_char):
            if char != '*':
                last_non_star_index = index
                
        if last_non_star_index != -1:
            output_strings.append("".join(t_list_char[:last_non_star_index+1]))
        else:
            output_strings.append("") # Should not happen based on problem description and samples

    for t in output_strings:
        print(t)

if __name__ == '__main__':
    solve()
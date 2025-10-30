def solve():
    s = input()
    t = input()
    if s == t:
        print(0)
        return
    
    current_s = list(s)
    target_t = list(t)
    intermediate_strings = []
    
    while "".join(current_s) != "".join(target_t):
        diff_indices = []
        for i in range(len(current_s)):
            if current_s[i] != target_t[i]:
                diff_indices.append(i)
        
        if not diff_indices:
            break
            
        next_strings = []
        for index_to_change in diff_indices:
            temp_s = list(current_s)
            temp_s[index_to_change] = target_t[index_to_change]
            next_strings.append("".join(temp_s))
            
        best_next_string = min(next_strings)
        intermediate_strings.append(best_next_string)
        current_s = list(best_next_string)
        
    print(len(intermediate_strings))
    for string in intermediate_strings:
        print(string)

if __name__ == '__main__':
    solve()
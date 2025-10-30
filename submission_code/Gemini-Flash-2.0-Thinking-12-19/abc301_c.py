def solve():
    s = input()
    t = input()
    r_set = set(['a', 't', 'c', 'o', 'd', 'e', 'r'])
    s_counts = {}
    t_counts = {}
    s_at_count = 0
    t_at_count = 0
    
    for char_s in s:
        if char_s == '@':
            s_at_count += 1
        else:
            s_counts[char_s] = s_counts.get(char_s, 0) + 1
            
    for char_t in t:
        if char_t == '@':
            t_at_count += 1
        else:
            t_counts[char_t] = t_counts.get(char_t, 0) + 1
            
    for char_code in range(ord('a'), ord('z') + 1):
        char = chr(char_code)
        if char not in r_set:
            if s_counts.get(char, 0) != t_counts.get(char, 0):
                print("No")
                return
                
    s_r_sum = 0
    t_r_sum = 0
    for char in r_set:
        s_r_sum += s_counts.get(char, 0)
        t_r_sum += t_counts.get(char, 0)
        
    if s_at_count + s_r_sum != t_at_count + t_r_sum:
        print("No")
        return
        
    l_sum = 0
    for char in r_set:
        l_sum += max(0, s_counts.get(char, 0) - t_counts.get(char, 0))
        
    if l_sum <= t_at_count:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()
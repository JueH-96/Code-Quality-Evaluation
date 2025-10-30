def solve():
    s = input()
    t = input()
    r_chars = set(['a', 't', 'c', 'o', 'd', 'e', 'r'])
    s_counts = {}
    t_counts = {}
    s_at_count = 0
    t_at_count = 0
    
    for char in s:
        if char == '@':
            s_at_count += 1
        else:
            s_counts[char] = s_counts.get(char, 0) + 1
            
    for char in t:
        if char == '@':
            t_at_count += 1
        else:
            t_counts[char] = t_counts.get(char, 0) + 1
            
    for char_code in range(ord('a'), ord('z') + 1):
        char = chr(char_code)
        if char not in r_chars:
            s_count = s_counts.get(char, 0)
            t_count = t_counts.get(char, 0)
            if s_count != t_count:
                print("No")
                return
                
    x_prime_sum = 0
    y_prime_sum = 0
    
    for char in r_chars:
        s_r_count = s_counts.get(char, 0)
        t_r_count = t_counts.get(char, 0)
        x_prime_sum += max(0, t_r_count - s_r_count)
        y_prime_sum += max(0, s_r_count - t_r_count)
        
    if x_prime_sum <= s_at_count and y_prime_sum <= t_at_count:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()
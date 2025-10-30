def solve():
    n, k = map(int, input().split())
    s = input()
    
    count = 0
    
    def is_good(t):
        while True:
            changed = False
            
            # Operation 1
            for i in range(len(t) - 1):
                if t[i] == t[i+1]:
                    t = t[:i] + t[i+2:]
                    changed = True
                    break
            
            if changed:
                continue
            
            # Operation 2
            if 'A' in t and 'B' in t and 'C' in t:
                a_idx = t.find('A')
                b_idx = t.find('B')
                c_idx = t.find('C')
                
                t_list = list(t)
                t_list.pop(max(a_idx, b_idx, c_idx))
                t_list.pop(max(min(a_idx, b_idx, c_idx), min(x for x in [a_idx, b_idx, c_idx] if x != max(a_idx, b_idx, c_idx))))
                t_list.pop(min(a_idx, b_idx, c_idx))
                
                t = "".join(t_list)
                changed = True
                continue
            
            if not changed:
                break
        
        return len(t) == 0
    
    def generate_strings(index, current_string):
        nonlocal count
        
        if index == n:
            good_substrings = 0
            for i in range(n):
                for j in range(i, n):
                    if is_good(current_string[i:j+1]):
                        good_substrings += 1
            
            if good_substrings >= k:
                count = (count + 1) % 998244353
            return
        
        if s[index] == '?':
            generate_strings(index + 1, current_string + 'A')
            generate_strings(index + 1, current_string + 'B')
            generate_strings(index + 1, current_string + 'C')
        else:
            generate_strings(index + 1, current_string + s[index])
            
    generate_strings(0, "")
    print(count)

solve()
def solve():
    s = input()
    t = input()
    s_len = len(s)
    t_len = len(t)
    correct_positions = []
    s_ptr = 0
    t_ptr = 0
    while s_ptr < s_len:
        while t_ptr < t_len:
            if s[s_ptr] == t[t_ptr]:
                correct_positions.append(t_ptr + 1)
                s_ptr += 1
                t_ptr += 1
                break
            else:
                t_ptr += 1
        else:
            # This should not happen based on problem description, but just in case.
            break 
            
    print(*(correct_positions))

if __name__ == '__main__':
    solve()
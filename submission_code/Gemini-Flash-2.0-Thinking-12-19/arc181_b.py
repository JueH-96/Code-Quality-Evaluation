def solve():
    s = input()
    x = input()
    y = input()
    len_s = len(s)
    n_x_0 = x.count('0')
    n_x_1 = x.count('1')
    n_y_0 = y.count('0')
    n_y_1 = y.count('1')
    a = n_x_1 - n_y_1
    b = n_y_0 - n_x_0
    if a == 0:
        if b == 0:
            return "Yes"
        else:
            return "No"
    else:
        len_t_val = (b * len_s) / a
        if len_t_val < 0 or len_t_val != int(len_t_val):
            return "No"
        len_t = int(len_t_val)
        t_chars = [None] * len_t
        possible = True
        n_x = len(x)
        n_y = len(y)
        len_fx = n_x_0 * len_s + n_x_1 * len_t
        len_fy = n_y_0 * len_s + n_y_1 * len_t
        if len_fx != len_fy:
            return "No"
        total_len = len_fx
        
        current_x_len = 0
        current_y_len = 0
        
        for k in range(total_len):
            block_index_x = -1
            block_index_y = -1
            pos_in_block_x = -1
            pos_in_block_y = -1
            
            current_block_start_x = 0
            for i in range(n_x):
                block_len_x = len_s if x[i] == '0' else len_t
                block_end_x = current_block_start_x + block_len_x
                if current_block_start_x <= k < block_end_x:
                    block_index_x = i
                    pos_in_block_x = k - current_block_start_x
                    break
                current_block_start_x = block_end_x
                
            current_block_start_y = 0
            for j in range(n_y):
                block_len_y = len_s if y[j] == '0' else len_t
                block_end_y = current_block_start_y + block_len_y
                if current_block_start_y <= k < block_end_y:
                    block_index_y = j
                    pos_in_block_y = k - current_block_start_y
                    break
                current_block_start_y = block_end_y
                
            char_x = ''
            if x[block_index_x] == '0':
                char_x = s[pos_in_block_x]
            else:
                if len_t > 0:
                    char_x = t_chars[pos_in_block_x]
                else:
                    char_x = ''
                    
            char_y = ''
            if y[block_index_y] == '0':
                char_y = s[pos_in_block_y]
            else:
                if len_t > 0:
                    char_y = t_chars[pos_in_block_y]
                else:
                    char_y = ''
                    
            if x[block_index_x] == '0' and y[block_index_y] == '0':
                if s[pos_in_block_x] != s[pos_in_block_y]:
                    possible = False
                    break
            elif x[block_index_x] == '0' and y[block_index_y] == '1':
                expected_char = s[pos_in_block_x]
                if len_t > 0:
                    if t_chars[pos_in_block_y] is None:
                        t_chars[pos_in_block_y] = expected_char
                    elif t_chars[pos_in_block_y] != expected_char:
                        possible = False
                        break
                else:
                    possible = False
                    break
            elif x[block_index_x] == '1' and y[block_index_y] == '0':
                expected_char = s[pos_in_block_y]
                if len_t > 0:
                    if t_chars[pos_in_block_x] is None:
                        t_chars[pos_in_block_x] = expected_char
                    elif t_chars[pos_in_block_x] != expected_char:
                        possible = False
                        break
                else:
                    possible = False
                    break
            elif x[block_index_x] == '1' and y[block_index_y] == '1':
                if len_t > 0:
                    if t_chars[pos_in_block_x] is not None and t_chars[pos_in_block_y] is not None:
                        if t_chars[pos_in_block_x] != t_chars[pos_in_block_y]:
                            possible = False
                            break
                    elif t_chars[pos_in_block_x] is not None:
                        t_chars[pos_in_block_y] = t_chars[pos_in_block_x]
                    elif t_chars[pos_in_block_y] is not None:
                        t_chars[pos_in_block_x] = t_chars[pos_in_block_y]
                        
            if not possible:
                break
                
        if not possible:
            return "No"
            
        t_str = "".join(t_chars) if len_t > 0 else ""
        
        fx_str = ""
        for char_code in x:
            if char_code == '0':
                fx_str += s
            else:
                fx_str += t_str
                
        fy_str = ""
        for char_code in y:
            if char_code == '0':
                fy_str += s
            else:
                fy_str += t_str
                
        if fx_str == fy_str:
            return "Yes"
        else:
            return "No"

t = int(input())
results = []
for _ in range(t):
    results.append(solve())

for result in results:
    print(result)
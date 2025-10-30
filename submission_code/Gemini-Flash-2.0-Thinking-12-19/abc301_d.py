def solve():
    s = input()
    n = int(input())
    s_len = len(s)
    
    def to_decimal(binary_str):
        return int(binary_str, 2) if binary_str else 0
        
    def get_min_val(s_pattern):
        temp_s = list(s_pattern)
        for i in range(len(temp_s)):
            if temp_s[i] == '?':
                temp_s[i] = '0'
        return to_decimal("".join(temp_s))
        
    min_val = get_min_val(s)
    if min_val > n:
        print(-1)
        return
        
    binary_n = bin(n)[2:]
    padded_binary_n = binary_n.zfill(s_len)
    
    result_binary_list = []
    
    for i in range(s_len):
        char_s = s[i]
        if char_s == '0':
            result_binary_list.append('0')
        elif char_s == '1':
            result_binary_list.append('1')
        elif char_s == '?':
            # Try to put '1'
            test_binary_list = list(result_binary_list)
            test_binary_list.append('1')
            for j in range(i + 1, s_len):
                if s[j] == '0':
                    test_binary_list.append('0')
                else:
                    test_binary_list.append('1')
            
            test_binary_str = "".join(test_binary_list)
            if len(test_binary_str) > s_len:
                test_binary_str = test_binary_str[-s_len:]
            elif len(test_binary_str) < s_len:
                test_binary_str = test_binary_str.zfill(s_len)
                
            if test_binary_str <= padded_binary_n:
                result_binary_list.append('1')
            else:
                result_binary_list.append('0')
                
    result_binary_str = "".join(result_binary_list)
    print(to_decimal(result_binary_str))

if __name__ == '__main__':
    solve()
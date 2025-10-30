def solve():
    s = input()
    n = int(input())
    
    def to_int(binary_string):
        if not binary_string:
            return 0
        return int(binary_string, 2)
        
    min_s_list = list(s)
    for i in range(len(min_s_list)):
        if min_s_list[i] == '?':
            min_s_list[i] = '0'
    min_val = to_int("".join(min_s_list))
    
    if min_val > n:
        print(-1)
        return
        
    result_str_list = []
    for i in range(len(s)):
        char = s[i]
        if char == '0':
            result_str_list.append('0')
        elif char == '1':
            result_str_list.append('1')
        elif char == '?':
            temp_result_list = list(result_str_list)
            temp_result_list.append('1')
            future_s_list = list(s[i+1:])
            for j in range(len(future_s_list)):
                if future_s_list[j] == '?':
                    future_s_list[j] = '1'
            test_s = "".join(temp_result_list) + "".join(future_s_list)
            test_val = to_int(test_s)
            if test_val <= n:
                result_str_list.append('1')
            else:
                result_str_list.append('0')
                
    result_val = to_int("".join(result_str_list))
    print(result_val)

if __name__ == '__main__':
    solve()
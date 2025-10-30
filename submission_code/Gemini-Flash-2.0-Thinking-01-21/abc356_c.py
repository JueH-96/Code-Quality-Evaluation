def solve():
    n, m, k = map(int, input().split())
    tests = []
    for _ in range(m):
        line = input().split()
        c_i = int(line[0])
        a_i = list(map(int, line[1:c_i+1]))
        r_i = line[c_i+1]
        tests.append((a_i, r_i))
    
    valid_combinations_count = 0
    for i in range(2**n):
        is_valid_combination = True
        key_status = []
        for bit_index in range(n):
            if (i >> bit_index) & 1:
                key_status.append(1) # real
            else:
                key_status.append(0) # dummy
                
        for test_keys, result in tests:
            real_keys_in_test = 0
            for key_index in test_keys:
                if key_status[key_index-1] == 1:
                    real_keys_in_test += 1
            
            if result == 'o':
                if real_keys_in_test < k:
                    is_valid_combination = False
                    break
            elif result == 'x':
                if real_keys_in_test >= k:
                    is_valid_combination = False
                    break
                    
        if is_valid_combination:
            valid_combinations_count += 1
            
    print(valid_combinations_count)

if __name__ == '__main__':
    solve()
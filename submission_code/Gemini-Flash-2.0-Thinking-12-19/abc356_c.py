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
        binary_combination = bin(i)[2:].zfill(n)
        is_valid_combination = True
        for test in tests:
            inserted_keys, result = test
            real_keys_inserted_count = 0
            for key_index in inserted_keys:
                if binary_combination[key_index-1] == '1':
                    real_keys_inserted_count += 1
            
            expected_result = 'o' if real_keys_inserted_count >= k else 'x'
            if expected_result != result:
                is_valid_combination = False
                break
        if is_valid_combination:
            valid_combinations_count += 1
            
    print(valid_combinations_count)

if __name__ == '__main__':
    solve()
def solve():
    n = int(input())
    s = input()
    max_len = 0
    for i in range(n):
        if s[i] == '/':
            current_max_len = 1
            max_len = max(max_len, current_max_len)
            for k in range(1, n):
                start_index = i - k
                end_index = i + k
                if start_index < 0 or end_index >= n:
                    break
                prefix = s[start_index:i]
                suffix = s[i+1:end_index+1]
                is_prefix_ones = True
                for char in prefix:
                    if char != '1':
                        is_prefix_ones = False
                        break
                is_suffix_twos = True
                for char in suffix:
                    if char != '2':
                        is_suffix_twos = False
                        break
                if is_prefix_ones and is_suffix_twos:
                    current_len = 2 * k + 1
                    max_len = max(max_len, current_len)
                else:
                    break
                    
    print(max_len)

if __name__ == '__main__':
    solve()
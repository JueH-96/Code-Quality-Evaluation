def solve():
    n = int(input())
    if n <= 10:
        print(n - 1)
        return
    
    cumulative_count = 10
    length = 2
    while True:
        count_for_length = 9 * (10 ** ((length - 1) // 2))
        if cumulative_count + count_for_length >= n:
            break
        cumulative_count += count_for_length
        length += 1
        
    m_index = n - cumulative_count + 9 * (10 ** ((length - 1) // 2))
    
    if length % 2 == 1:
        m = (length - 1) // 2
        prefix_val = 10**m + (m_index - 1)
        prefix_str = str(prefix_val)
        first_m_digits = prefix_str[:m]
        reversed_m_digits = first_m_digits[::-1]
        result_str = prefix_str + reversed_m_digits
        print(result_str)
    else:
        m = length // 2
        prefix_val = 10**(m-1) + (m_index - 1)
        prefix_str = str(prefix_val)
        reversed_prefix_str = prefix_str[::-1]
        result_str = prefix_str + reversed_prefix_str
        print(result_str)

if __name__ == '__main__':
    solve()
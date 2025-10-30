def solve():
    n_input = int(input())
    if n_input == 1:
        print(0)
        return
    n = n_input - 1
    palindrome_lengths = []
    counts = []
    cumulative_counts = []
    current_cumulative_count = 0
    length = 1
    while True:
        count = 9 * (10**((length - 1) // 2))
        if length == 1:
            count = 9
        else:
            count = 9 * (10**((length - 1) // 2))
        counts.append(count)
        current_cumulative_count += count
        cumulative_counts.append(current_cumulative_count)
        palindrome_lengths.append(length)
        if current_cumulative_count >= n:
            target_length = length
            break
        length += 1
        
    index_in_length = n
    if target_length > 1:
        index_in_length = n - cumulative_counts[palindrome_lengths.index(target_length) - 1]
    else:
        index_in_length = n
        
    index_in_length -= 1
    
    m = (target_length + 1) // 2
    prefix_digits = []
    temp_index = index_in_length
    
    for i in range(m):
        if i == 0:
            digit = (temp_index // (10**(m - 1 - i)))
            digit += 1
        else:
            digit = (temp_index // (10**(m - 1 - i)))
        prefix_digits.append(digit)
        temp_index %= (10**(m - 1 - i))
        
    prefix_str = "".join(map(str, prefix_digits))
    reversed_part = ""
    if target_length % 2 == 0:
        reversed_part_digits = prefix_digits[:m][::-1]
        reversed_part = "".join(map(str, reversed_part_digits))
    else:
        reversed_part_digits = prefix_digits[:m-1][::-1]
        reversed_part = "".join(map(str, reversed_part_digits))
        
    palindrome_str = prefix_str + reversed_part
    print(int(palindrome_str))

if __name__ == '__main__':
    solve()
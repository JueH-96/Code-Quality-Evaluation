def is_good_string(s):
    count_a = s.count('A')
    count_b = s.count('B')
    count_c = s.count('C')
    return (count_a % 2 == count_b % 2) and (count_a % 2 == count_c % 2)

def solve():
    n, k = map(int, input().split())
    s_template = input()
    q_indices = [i for i, char in enumerate(s_template) if char == '?']
    num_q = len(q_indices)
    count_ways = 0
    mod = 998244353
    
    replacements = ['A', 'B', 'C']
    
    for i in range(3**num_q):
        current_s_list = list(s_template)
        temp_i = i
        for j in range(num_q):
            replacement_index = temp_i % 3
            current_s_list[q_indices[j]] = replacements[replacement_index]
            temp_i //= 3
        
        current_s = "".join(current_s_list)
        good_substring_count = 0
        for start_index in range(n):
            for end_index in range(start_index, n):
                substring = current_s[start_index:end_index+1]
                if is_good_string(substring):
                    good_substring_count += 1
                    
        if good_substring_count >= k:
            count_ways += 1
            
    print(count_ways % mod)

if __name__ == '__main__':
    solve()
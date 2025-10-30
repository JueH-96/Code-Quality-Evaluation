def is_good_string(s):
    counts = {'A': 0, 'B': 0, 'C': 0}
    for char in s:
        if char in counts:
            counts[char] += 1
    n_a, n_b, n_c = counts['A'], counts['B'], counts['C']
    m = min(n_a, n_b, n_c)
    n_a -= m
    n_b -= m
    n_c -= m
    return n_a % 2 == 0 and n_b % 2 == 0 and n_c % 2 == 0

def solve():
    n, k = map(int, input().split())
    s = input()
    
    memo = {}
    
    def get_good_substring_count(full_string):
        count = 0
        length = len(full_string)
        for i in range(length):
            for j in range(i, length):
                substring = full_string[i:j+1]
                if is_good_string(substring):
                    count += 1
        return count
        
    def generate_strings(index, current_string):
        if index == n:
            good_substrings_count = get_good_substring_count(current_string)
            if good_substrings_count >= k:
                return 1
            else:
                return 0
        
        if (index, current_string) in memo:
            return memo[(index, current_string)]
            
        count_sum = 0
        char_at_index = s[index]
        possible_chars = []
        if char_at_index == '?':
            possible_chars = ['A', 'B', 'C']
        else:
            possible_chars = [char_at_index]
            
        for char in possible_chars:
            count_sum = (count_sum + generate_strings(index + 1, current_string + char)) % 998244353
            
        memo[(index, current_string)] = count_sum
        return count_sum
        
    result = generate_strings(0, "")
    print(result)

if __name__ == '__main__':
    solve()
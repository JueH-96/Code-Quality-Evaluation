def solve():
    MOD = 998244353
    n, k = map(int, input().split())
    s = input().strip()
    
    memo = {}
    
    def count_good_substrings(string):
        n = len(string)
        prefix_a_parity = [0] * (n + 1)
        prefix_b_parity = [0] * (n + 1)
        prefix_c_parity = [0] * (n + 1)
        
        for i in range(n):
            prefix_a_parity[i+1] = prefix_a_parity[i] ^ (1 if string[i] == 'A' else 0)
            prefix_b_parity[i+1] = prefix_b_parity[i] ^ (1 if string[i] == 'B' else 0)
            prefix_c_parity[i+1] = prefix_c_parity[i] ^ (1 if string[i] == 'C' else 0)
        
        good_count = 0
        for i in range(n):
            for j in range(i, n):
                parity_a = prefix_a_parity[j+1] ^ prefix_a_parity[i]
                parity_b = prefix_b_parity[j+1] ^ prefix_b_parity[i]
                parity_c = prefix_c_parity[j+1] ^ prefix_c_parity[i]
                
                if parity_a == parity_b == parity_c:
                    good_count += 1
        
        return good_count
    
    def replace_and_count(index, s_list):
        if index == n:
            return 1 if count_good_substrings(''.join(s_list)) >= k else 0
        
        key = (index, ''.join(s_list))
        if key in memo:
            return memo[key]
        
        if s_list[index] != '?':
            result = replace_and_count(index+1, s_list)
        else:
            result = 0
            for c in ['A', 'B', 'C']:
                s_list[index] = c
                result = (result + replace_and_count(index+1, s_list)) % MOD
                s_list[index] = '?'  # backtrack
        
        memo[key] = result
        return result
    
    return replace_and_count(0, list(s))

print(solve())
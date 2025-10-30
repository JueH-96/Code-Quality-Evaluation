from bisect import bisect_left, bisect_right

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    m = int(data[1])
    a = list(map(int, data[2:2+n]))
    
    S = [0] * (n + 1)
    for i in range(1, n + 1):
        S[i] = S[i - 1] + a[i - 1]
    
    # Precompute s_minus_1_indices and s_values for each r
    s_minus_1_indices = [[] for _ in range(m)]
    s_values = [[] for _ in range(m)]
    
    for i in range(n):
        r = S[i] % m
        s_minus_1_indices[r].append(i)
        s_values[r].append(S[i])
    
    # Sort the lists for binary search
    for r in range(m):
        s_minus_1_indices[r].sort()
        s_values[r].sort()
    
    total = 0
    
    for t in range(1, n + 1):
        S_t_minus_1 = S[t - 1]
        T = S[-1]
        
        # Condition A: D mod M == 0 and D <= T/2
        r_a = S_t_minus_1 % m
        s_minus_1_list = s_minus_1_indices[r_a]
        s_value_list = s_values[r_a]
        
        # Calculate low for condition A
        low = S_t_minus_1 - T / 2
        
        # Find the left index in s_value_list
        left = bisect_left(s_value_list, low)
        
        # Find the right index in s_minus_1_list
        if not s_minus_1_list:
            right = -1
        else:
            right = bisect_right(s_minus_1_list, t - 2) - 1
        
        if left <= right and left < len(s_value_list):
            condition_A_count = right - left + 1
        else:
            condition_A_count = 0
        
        # Condition B: D >= T - D and D mod M == 0
        r_b = (S_t_minus_1 - T) % m
        s_minus_1_list_b = s_minus_1_indices[r_b]
        s_value_list_b = s_values[r_b]
        
        # Calculate low_b for condition B
        low_b = S_t_minus_1 - T / 2
        
        # Find the left_b index in s_value_list_b
        if not s_value_list_b:
            left_b = -1
        else:
            left_b = bisect_right(s_value_list_b, low_b) - 1
        
        # Find the right_b index in s_minus_1_list_b
        if not s_minus_1_list_b:
            right_b = -1
        else:
            right_b = bisect_right(s_minus_1_list_b, t - 2) - 1
        
        if left_b <= right_b and left_b < len(s_value_list_b):
            condition_B_count = right_b - left_b + 1
        else:
            condition_B_count = 0
        
        total += condition_A_count + condition_B_count
    
    print(total)

if __name__ == '__main__':
    main()
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    s = input()
    
    # Precompute the number of 'X' characters with each value c for each position
    x_count_suffix = [[0, 0, 0] for _ in range(n+1)]
    for i in range(n-1, -1, -1):
        for c in range(3):
            x_count_suffix[i][c] = x_count_suffix[i+1][c]
        if s[i] == 'X':
            x_count_suffix[i][a[i]] += 1
    
    result = 0
    
    m_positions = [i for i in range(n) if s[i] == 'M']
    e_positions = [i for i in range(n) if s[i] == 'E']
    
    for i in m_positions:
        for j in e_positions:
            if j <= i:
                continue
            
            present_pair = {a[i], a[j]}
            
            # Count the number of 'X' characters with each value after position j
            x_counts = x_count_suffix[j+1]
            
            for c in range(3):
                present = present_pair | {c}
                
                if 0 not in present:
                    mex_value = 0
                elif 1 not in present:
                    mex_value = 1
                elif 2 not in present:
                    mex_value = 2
                else:
                    mex_value = 3
                
                result += mex_value * x_counts[c]
    
    return result

print(solve())
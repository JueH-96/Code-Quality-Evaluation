import math

def get_square_free_part(n):
    if n == 0:
        return 1 # or 0, doesn't matter as long as consistent
    square_free_part = n
    i = 2
    while i * i <= square_free_part:
        if square_free_part % (i * i) == 0:
            square_free_part //= (i * i)
        else:
            i += 1
    return square_free_part

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    zero_indices = []
    non_zero_values_square_free_parts = []
    
    for i in range(n):
        if a[i] == 0:
            zero_indices.append(i+1)
        else:
            non_zero_values_square_free_parts.append(get_square_free_part(a[i]))
            
    count = 0
    for i_index in zero_indices:
        count += (n - i_index)
        
    square_free_counts = {}
    for r in non_zero_values_square_free_parts:
        square_free_counts[r] = square_free_counts.get(r, 0) + 1
        
    for r in square_free_counts:
        m_r = square_free_counts[r]
        if m_r >= 2:
            count += (m_r * (m_r - 1)) // 2
            
    print(count)

if __name__ == '__main__':
    solve()
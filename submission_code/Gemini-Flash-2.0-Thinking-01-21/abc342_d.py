import math

def get_square_free_part(n):
    if n == 0:
        return 0
    square_free = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            count = 0
            while n % d == 0:
                n //= d
                count += 1
            if count % 2 == 1:
                square_free *= d
        d += 1
    if n > 1:
        square_free *= n
    return square_free

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    zero_indices = []
    non_zero_values = []
    for i in range(n):
        if a[i] == 0:
            zero_indices.append(i+1)
        else:
            non_zero_values.append(a[i])
            
    count_zeros = 0
    for index in zero_indices:
        count_zeros += (n - index + 1)
        
    square_free_parts = []
    for val in non_zero_values:
        square_free_parts.append(get_square_free_part(val))
        
    square_free_counts = {}
    for part in square_free_parts:
        square_free_counts[part] = square_free_counts.get(part, 0) + 1
        
    count_square_free = 0
    for part in square_free_counts:
        count = square_free_counts[part]
        count_square_free += count * (count - 1) // 2
        
    total_pairs = count_zeros + count_square_free
    print(total_pairs)

if __name__ == '__main__':
    solve()
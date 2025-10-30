def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    
    prefix_sums = [0] * (n + 1)
    for i in range(n):
        prefix_sums[i+1] = prefix_sums[i] + a[i]
        
    prefix_remainders = [0] * n
    for i in range(n):
        prefix_remainders[i] = prefix_sums[i+1] % m
        
    total_sum_mod_m = prefix_sums[n] % m
    r_prime = (m - total_sum_mod_m) % m
    
    remainder_indices = [[] for _ in range(m)]
    for i in range(n):
        remainder_indices[prefix_remainders[i]].append(i)
        
    count = 0
    for x in range(m):
        c_x = len(remainder_indices[x])
        count += c_x * (c_x - 1) // 2
        
    for x in range(m):
        y = (x + r_prime) % m
        indices_x = remainder_indices[x]
        indices_y = remainder_indices[y]
        for i_x in indices_x:
            for i_y in indices_y:
                if i_y < i_x:
                    count += 1
                    
    print(count)

if __name__ == '__main__':
    solve()
def solve():
    n, m = map(int, input().split())
    a_sequences = []
    for _ in range(n):
        a_sequences.append(list(map(int, input().split())))
    
    total_f_sum = 0
    mod_val = 998244353
    
    for i in range(n):
        for j in range(i, n):
            u = list(a_sequences[i])
            v = list(a_sequences[j])
            f_val = 0
            found_x = False
            for x in range(m + 1):
                if u == v:
                    f_val = x
                    found_x = True
                    break
                
                next_u = [0] * m
                current_sum_u = 0
                for k in range(m):
                    current_sum_u = (current_sum_u + u[k]) % 2
                    next_u[k] = current_sum_u
                u = next_u[:]
                
                next_v = [0] * m
                current_sum_v = 0
                for k in range(m):
                    current_sum_v = (current_sum_v + v[k]) % 2
                    next_v[k] = current_sum_v
                v = next_v[:]
                
            if found_x:
                total_f_sum = (total_f_sum + f_val) % mod_val
                
    print(total_f_sum)

if __name__ == '__main__':
    solve()
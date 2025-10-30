def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    queries = []
    for _ in range(q):
        l, r, L, R = map(int, input().split())
        queries.append(((l, r), (L, R)))
    
    results = []
    for (l_range, r_range) in queries:
        l_i, r_i = l_range
        L_i, R_i = r_range
        len_a_sub = r_i - l_i + 1
        len_b_sub = R_i - L_i + 1
        if len_a_sub != len_b_sub:
            results.append("No")
        else:
            count_a = [0] * (n + 1)
            count_b = [0] * (n + 1)
            for i in range(l_i - 1, r_i):
                count_a[a[i]] += 1
            for i in range(L_i - 1, R_i):
                count_b[b[i]] += 1
            
            possible = True
            for i in range(1, n + 1):
                if count_a[i] != count_b[i]:
                    possible = False
                    break
            if possible:
                results.append("Yes")
            else:
                results.append("No")
                
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()
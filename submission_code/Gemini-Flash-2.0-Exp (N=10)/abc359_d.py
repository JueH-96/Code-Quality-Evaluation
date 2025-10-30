def solve():
    n, k = map(int, input().split())
    s = input()
    q_count = s.count('?')
    
    ans = 0
    mod = 998244353

    for i in range(2**q_count):
        temp_s = list(s)
        temp_bin = bin(i)[2:].zfill(q_count)
        q_idx = 0
        for j in range(n):
            if temp_s[j] == '?':
                temp_s[j] = 'A' if temp_bin[q_idx] == '0' else 'B'
                q_idx += 1
        
        temp_s = "".join(temp_s)
        
        is_good = True
        for j in range(n - k + 1):
            sub = temp_s[j:j+k]
            if sub == sub[::-1]:
                is_good = False
                break
        
        if is_good:
            ans = (ans + 1) % mod
    
    print(ans)

solve()
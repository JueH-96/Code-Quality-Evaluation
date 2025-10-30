def solve():
    n, k = map(int, input().split())
    s = input()
    
    blocks = []
    l = -1
    for i in range(n):
        if s[i] == '1' and (l == -1 or s[i-1] == '0'):
            l = i
        if s[i] == '1' and (i == n - 1 or s[i+1] == '0'):
            blocks.append((l, i))
            l = -1
    
    l_k_minus_1, r_k_minus_1 = blocks[k-2]
    l_k, r_k = blocks[k-1]
    
    t = ""
    for i in range(r_k_minus_1 + 1):
        t += s[i]
    
    for i in range(r_k - l_k + 1):
        t += '1'
        
    for i in range(r_k_minus_1 + 1 + (r_k - l_k + 1), r_k + 1):
        t += '0'
        
    for i in range(r_k + 1, n):
        t += s[i]
        
    print(t)

solve()
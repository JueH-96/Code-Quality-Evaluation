def solve():
    n, k = map(int, input().split())
    s = input()
    
    q_indices = [i for i, char in enumerate(s) if char == '?']
    q = len(q_indices)
    
    count = 0
    
    for i in range(2**q):
        temp_s = list(s)
        binary = bin(i)[2:].zfill(q)
        
        for j in range(q):
            if binary[j] == '0':
                temp_s[q_indices[j]] = 'A'
            else:
                temp_s[q_indices[j]] = 'B'
        
        temp_s = "".join(temp_s)
        
        is_good = True
        for j in range(n - k + 1):
            sub = temp_s[j:j+k]
            if sub == sub[::-1]:
                is_good = False
                break
        
        if is_good:
            count = (count + 1) % 998244353
    
    print(count)

solve()
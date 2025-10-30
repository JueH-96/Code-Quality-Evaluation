def solve():
    s = input()
    n = len(s)
    q_count = s.count('?')
    mod = 998244353
    ans = 0

    for i in range(2**q_count):
        temp = bin(i)[2:].zfill(q_count)
        cur = ""
        q_idx = 0
        for char in s:
            if char == '?':
                cur += '(' if temp[q_idx] == '0' else ')'
                q_idx += 1
            else:
                cur += char
        
        balance = 0
        valid = True
        for char in cur:
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            if balance < 0:
                valid = False
                break
        if valid and balance == 0:
            ans = (ans + 1) % mod
    
    print(ans)

solve()
# YOUR CODE HERE
MOD = 998244353

def solve():
    s = input()
    n = len(s)
    q_count = s.count('?')
    
    ans = 0
    
    for i in range(52**q_count):
        temp = list(s)
        k = i
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        
        for j in range(n):
            if temp[j] == '?':
                idx = k % 52
                temp[j] = chars[idx]
                k //= 52
        
        cur_str = "".join(temp)
        is_ok = True
        
        for j in range(n - 3):
            sub = cur_str[j:j+4]
            if len(sub) == 4:
                if sub[0].isupper() and sub[1].isupper() and sub[2].islower() and sub[3].isupper() and sub[0] == sub[1]:
                    is_ok = False
                    break
        if is_ok:
            ans +=1

    print(ans % MOD)

solve()
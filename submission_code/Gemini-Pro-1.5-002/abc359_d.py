# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    s = input()
    mod = 998244353

    q_count = s.count('?')
    
    ans = 0
    for i in range(2**q_count):
        temp = bin(i)[2:].zfill(q_count)
        curr_s = ""
        temp_idx = 0
        for char in s:
            if char == '?':
                curr_s += 'A' if temp[temp_idx] == '0' else 'B'
                temp_idx += 1
            else:
                curr_s += char
        
        is_good = True
        for j in range(n - k + 1):
            sub = curr_s[j:j+k]
            if sub == sub[::-1]:
                is_good = False
                break
        
        if is_good:
            ans = (ans + 1) % mod
            
    print(ans)

solve()
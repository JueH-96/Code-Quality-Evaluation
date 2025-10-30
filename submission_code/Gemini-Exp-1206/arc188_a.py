def is_good(s):
    a_count = s.count('A')
    b_count = s.count('B')
    c_count = s.count('C')
    
    if (a_count + b_count + c_count) % 2 != 0:
        return False

    min_count = min(a_count, b_count, c_count)
    
    
    return (a_count - min_count) % 2 == 0 and (b_count - min_count) % 2 == 0 and (c_count - min_count) % 2 == 0

def solve():
    n, k = map(int, input().split())
    s = input()
    q_count = s.count('?')
    ans = 0
    mod = 998244353

    for i in range(3**q_count):
        temp = bin(i)[2:].zfill(q_count * 2)
        
        
        curr = ""
        q_idx = 0
        for char in s:
            if char == '?':
                
                
                if temp[q_idx*2:q_idx*2+2] == '00':
                    curr += 'A'
                elif temp[q_idx*2:q_idx*2+2] == '01':
                    curr += 'B'
                elif temp[q_idx*2:q_idx*2+2] == '10':
                    curr += 'C'
                else:
                    curr += 'A'
                q_idx += 1
            else:
                curr += char
        
        good_count = 0
        for j in range(n):
            for l in range(j, n):
                if is_good(curr[j:l+1]):
                    good_count += 1
        
        if good_count >= k:
            ans = (ans + 1) % mod

    print(ans)

solve()
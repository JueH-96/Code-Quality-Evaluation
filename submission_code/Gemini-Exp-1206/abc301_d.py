def solve():
    s = input()
    n = int(input())
    
    q_count = s.count('?')
    ans = -1
    
    for i in range(1 << q_count):
        temp = bin(i)[2:].zfill(q_count)
        cur = ""
        temp_idx = 0
        for char in s:
            if char == '?':
                cur += temp[temp_idx]
                temp_idx += 1
            else:
                cur += char
        
        cur_val = int(cur, 2)
        if cur_val <= n:
            ans = max(ans, cur_val)
            
    print(ans)

solve()
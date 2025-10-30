def solve():
    n = int(input())
    s = input()
    
    max_len = 0
    for i in range(n):
        for j in range(i, n):
            sub = s[i:j+1]
            sub_len = len(sub)
            
            if sub_len % 2 == 0:
                continue
            
            mid = (sub_len - 1) // 2
            
            if sub[mid] != '/':
                continue
            
            is_1122 = True
            for k in range(mid):
                if sub[k] != '1':
                    is_1122 = False
                    break
            
            if not is_1122:
                continue
                
            for k in range(mid + 1, sub_len):
                if sub[k] != '2':
                    is_1122 = False
                    break
                    
            if is_1122:
                max_len = max(max_len, sub_len)
                
    print(max_len)

solve()
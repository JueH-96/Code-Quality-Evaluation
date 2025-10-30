def solve():
    n = int(input())
    s = input()
    
    max_len = 0
    
    for i in range(n):
        for j in range(i, n):
            sub = s[i:j+1]
            len_sub = len(sub)
            
            if len_sub % 2 != 0:
                mid = (len_sub + 1) // 2
                
                is_1122 = True
                
                for k in range(mid - 1):
                    if k >= len_sub or sub[k] != '1':
                        is_1122 = False
                        break
                
                if mid - 1 >= len_sub or sub[mid - 1] != '/':
                    is_1122 = False
                
                for k in range(mid, len_sub):
                    if k >= len_sub or sub[k] != '2':
                        is_1122 = False
                        break
                
                if is_1122:
                    max_len = max(max_len, len_sub)
    
    print(max_len)

solve()
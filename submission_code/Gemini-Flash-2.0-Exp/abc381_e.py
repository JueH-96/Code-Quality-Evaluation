def solve():
    n, q = map(int, input().split())
    s = input()
    
    for _ in range(q):
        l, r = map(int, input().split())
        sub = s[l-1:r]
        
        max_len = 0
        for i in range(1 << len(sub)):
            temp = ""
            for j in range(len(sub)):
                if (i >> j) & 1:
                    temp += sub[j]
            
            if not temp:
                continue
            
            len_t = len(temp)
            if len_t % 2 == 0:
                continue
            
            mid = (len_t + 1) // 2
            
            is_valid = True
            for k in range(mid - 1):
                if temp[k] != '1':
                    is_valid = False
                    break
            
            if temp[mid-1] != '/':
                is_valid = False
            
            for k in range(mid, len_t):
                if temp[k] != '2':
                    is_valid = False
                    break
            
            if is_valid:
                max_len = max(max_len, len_t)
        
        print(max_len)

solve()
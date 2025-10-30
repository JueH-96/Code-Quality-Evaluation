def solve():
    n, q = map(int, input().split())
    s = input()
    
    for _ in range(q):
        l, r = map(int, input().split())
        sub = s[l-1:r]
        
        max_len = 0
        
        for i in range(1 << len(sub)):
            curr = ""
            for j in range(len(sub)):
                if (i >> j) & 1:
                    curr += sub[j]
            
            if len(curr) % 2 == 0 or len(curr) == 0:
                continue
            
            mid = (len(curr) + 1) // 2
            
            is_1122 = True
            
            for j in range(mid - 1):
                if curr[j] != '1':
                    is_1122 = False
                    break
            
            if curr[mid - 1] != '/':
                is_1122 = False
            
            for j in range(mid, len(curr)):
                if curr[j] != '2':
                    is_1122 = False
                    break
            
            if is_1122:
                max_len = max(max_len, len(curr))
        
        print(max_len)

solve()
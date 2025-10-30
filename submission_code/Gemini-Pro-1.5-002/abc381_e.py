def solve():
    n, q = map(int, input().split())
    s = input()
    
    for _ in range(q):
        l, r = map(int, input().split())
        l -= 1
        
        sub = s[l:r]
        max_len = 0
        
        for length in range(1, len(sub) + 1, 2):
            for i in range(1 << len(sub)):
                temp = ""
                for j in range(len(sub)):
                    if (i >> j) & 1:
                        temp += sub[j]

                if len(temp) == length:
                    is_1122 = True
                    if len(temp) % 2 == 0:
                        is_1122 = False
                    else:
                        mid = (len(temp) + 1) // 2 - 1
                        if temp[mid] != '/':
                            is_1122 = False
                        else:
                            for k in range(mid):
                                if temp[k] != '1':
                                    is_1122 = False
                                    break
                            for k in range(mid + 1, len(temp)):
                                if temp[k] != '2':
                                    is_1122 = False
                                    break
                    
                    if is_1122:
                        max_len = max(max_len, length)
        
        print(max_len)

solve()
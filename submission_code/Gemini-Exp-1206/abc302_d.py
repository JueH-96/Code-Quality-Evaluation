def solve():
    n, m, d = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    a.sort()
    b.sort()
    
    ans = -1
    
    for i in range(n):
        left = 0
        right = m - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if abs(a[i] - b[mid]) <= d:
                ans = max(ans, a[i] + b[mid])
                
                
                left_temp = left
                right_temp = mid -1
                while left_temp <= right_temp:
                    mid_temp = (left_temp + right_temp) // 2
                    if abs(a[i] - b[mid_temp]) <= d:
                        ans = max(ans, a[i] + b[mid_temp])
                        right_temp = mid_temp - 1
                    else:
                        left_temp = mid_temp + 1
                        
                left_temp = mid + 1
                right_temp = right
                while left_temp <= right_temp:
                    mid_temp = (left_temp + right_temp) // 2
                    if abs(a[i] - b[mid_temp]) <= d:
                        ans = max(ans, a[i] + b[mid_temp])
                        left_temp = mid_temp + 1
                    else:
                        right_temp = mid_temp - 1
                
                
                break
            elif a[i] - b[mid] > d:
                right = mid - 1
            else:
                left = mid + 1
                
    print(ans)

solve()
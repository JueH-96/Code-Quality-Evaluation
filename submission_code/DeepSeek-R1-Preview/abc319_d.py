n, m = map(int, input().split())
L = list(map(int, input().split()))

max_L = max(L)
sum_L = sum(L)

if m >= n:
    print(max_L)
else:
    low = max_L
    high = sum_L + (n - 1)
    
    while low < high:
        mid = (low + high) // 2
        lines = 1
        current = 0
        possible = True
        
        for word in L:
            if current == 0:
                current = word
            else:
                temp = current + 1 + word
                if temp > mid:
                    lines += 1
                    current = word
                    if lines > m:
                        possible = False
                        break
                else:
                    current = temp
        
        if possible:
            high = mid
        else:
            low = mid + 1
    
    print(low)
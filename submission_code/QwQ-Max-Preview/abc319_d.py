n, m = map(int, input().split())
l = list(map(int, input().split()))

left = max(l)
right = sum(l) + (n - 1)
ans = right

while left <= right:
    mid = (left + right) // 2
    lines = 1
    current_sum = 0
    current_count = 0
    possible = True
    
    for word in l:
        if current_count == 0:
            required = word
        else:
            required = current_sum + word + current_count
        
        if required > mid:
            lines += 1
            current_sum = word
            current_count = 1
            if lines > m:
                possible = False
                break
        else:
            current_sum += word
            current_count += 1
    
    if possible:
        ans = mid
        right = mid - 1
    else:
        left = mid + 1

print(ans)
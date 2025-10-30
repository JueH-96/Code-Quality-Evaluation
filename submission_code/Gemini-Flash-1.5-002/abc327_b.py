# YOUR CODE HERE
B = int(input())
left = 1
right = 1000000
ans = -1
while left <= right:
    mid = (left + right) // 2
    val = 1
    flag = True
    try:
        val = mid**mid
    except OverflowError:
        flag = False
    if flag:
        if val == B:
            ans = mid
            break
        elif val < B:
            left = mid + 1
        else:
            right = mid - 1
    else:
        right = mid -1

print(ans)
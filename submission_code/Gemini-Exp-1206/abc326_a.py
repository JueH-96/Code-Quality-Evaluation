x, y = map(int, input().split())
diff = y - x
if diff > 0:
    if diff <= 2:
        print('Yes')
    else:
        print('No')
else:
    if diff >= -3:
        print('Yes')
    else:
        print('No')
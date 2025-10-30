H = int(input())
i = 0
height = 0
while height <= H:
    height += 2**i
    i += 1
print(i)
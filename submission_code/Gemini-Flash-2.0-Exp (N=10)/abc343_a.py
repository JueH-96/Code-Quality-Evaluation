a, b = map(int, input().split())
sum_ab = a + b
for i in range(10):
    if i != sum_ab:
        print(i)
        break
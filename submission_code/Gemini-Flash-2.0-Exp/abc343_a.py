A, B = map(int, input().split())
sum_val = A + B
for i in range(10):
    if i != sum_val:
        print(i)
        break
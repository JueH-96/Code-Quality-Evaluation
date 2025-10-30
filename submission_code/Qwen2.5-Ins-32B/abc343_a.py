a, b = map(int, input().split())
sum_ab = a + b
# Print any number between 0 and 9 that is not equal to sum_ab
for i in range(10):
    if i != sum_ab:
        print(i)
        break
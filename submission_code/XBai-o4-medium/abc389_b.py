x = int(input())
product = 1
n = 1
while True:
    n += 1
    product *= n
    if product == x:
        print(n)
        break
    if product > x:
        break
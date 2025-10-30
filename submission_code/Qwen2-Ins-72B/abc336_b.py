n = int(input())
res = 0
while n % 2 == 0 and n != 0:
    res += 1
    n //= 2
print(res)
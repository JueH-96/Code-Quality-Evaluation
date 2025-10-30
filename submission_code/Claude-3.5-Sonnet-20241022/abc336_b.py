N = int(input())
count = 0

while N > 0 and N % 2 == 0:
    count += 1
    N //= 2

print(count)
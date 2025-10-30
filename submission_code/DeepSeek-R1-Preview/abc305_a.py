n = int(input())
r = n % 5
if r == 0:
    print(n)
elif r <= 2:
    print(n - r)
else:
    print(n + (5 - r))
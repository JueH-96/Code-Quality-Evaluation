n = int(input())
q, r = divmod(n, 5)
if r <= 2:
    print(n - r)
else:
    print(n + (5 - r))
n, c1, c2 = input().split()
n = int(n)
s = input().strip()

result = ''.join([c if c == c1 else c2 for c in s])
print(result)
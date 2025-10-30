n, c1, c2 = input().split()
n = int(n)
s = input().strip()

result = ''.join(c2 if char != c1 else char for char in s)
print(result)
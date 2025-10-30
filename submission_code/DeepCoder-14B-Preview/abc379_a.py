n = int(input())
s = str(n)
a = int(s[0])
b = int(s[1])
c = int(s[2])

n1 = b * 100 + c * 10 + a
n2 = c * 100 + a * 10 + b

print(n1, n2)
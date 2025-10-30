n = int(input())
s = input().strip()
result = ''.join(c * 2 for c in s)
print(result)
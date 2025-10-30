n = int(input())
s = input().strip()

result = ''

for c in s:
    result += c + c

print(result)
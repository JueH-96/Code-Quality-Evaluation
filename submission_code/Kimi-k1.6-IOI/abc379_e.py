n = int(input())
s = input().strip()

result = 0
multiplier = 0

for j in reversed(range(n)):
    digit = int(s[j])
    multiplier = multiplier * 10 + 1
    result += digit * (j + 1) * multiplier

print(result)
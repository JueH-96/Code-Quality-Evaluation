n = int(input())

result = ""
for i in range(n):
    result += str(i % 2)
result += "1"

print(result)
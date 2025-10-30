n = int(input())
result = ""
for i in range(1, n + 1):
    if i % 3 == 0:
        result += "x"
    else:
        result += "o"
print(result)
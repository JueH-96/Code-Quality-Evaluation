s = input()
result = []
for char in s:
    result.append(char)
    while len(result) >= 3 and "".join(result[-3:]) == "ABC":
        result = result[:-3]
print("".join(result))
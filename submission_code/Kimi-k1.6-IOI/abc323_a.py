s = input().strip()
result = "Yes"
for i in range(1, 16, 2):
    if s[i] != '0':
        result = "No"
        break
print(result)
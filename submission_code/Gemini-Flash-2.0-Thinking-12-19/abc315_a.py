s = input()
vowels = set(['a', 'e', 'i', 'o', 'u'])
result = ""
for char in s:
    if char not in vowels:
        result += char
print(result)
s = input()
vowels = "aeiou"
result = ""
for char in s:
    if char not in vowels:
        result += char
print(result)
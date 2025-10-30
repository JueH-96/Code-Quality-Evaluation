S = input()
vowels = "aeiou"
result = ""
for char in S:
    if char not in vowels:
        result += char
print(result)
s = input()
vowels = "aeiou"
result_chars = [char for char in s if char not in vowels]
print("".join(result_chars))
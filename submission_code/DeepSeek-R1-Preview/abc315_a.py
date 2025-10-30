s = input().strip()
vowels = {'a', 'e', 'i', 'o', 'u'}
result = ''.join([char for char in s if char not in vowels])
print(result)
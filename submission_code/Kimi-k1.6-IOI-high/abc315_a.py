s = input().strip()
vowels = {'a', 'e', 'i', 'o', 'u'}
result = [char for char in s if char not in vowels]
print(''.join(result))
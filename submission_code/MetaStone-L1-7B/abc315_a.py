s = input().strip()
vowels = {'a', 'e', 'i', 'o', 'u'}
result = [c for c in s if c not in vowels]
print(''.join(result))
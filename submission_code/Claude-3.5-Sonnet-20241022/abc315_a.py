s = input()
vowels = 'aeiou'
result = ''.join(c for c in s if c not in vowels)
print(result)
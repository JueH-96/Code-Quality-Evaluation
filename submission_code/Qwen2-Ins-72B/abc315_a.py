S = input()
vowels = 'aeiou'
result = ''.join([c for c in S if c not in vowels])
print(result)
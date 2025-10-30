# YOUR CODE HERE
s = input().strip()
vowels = {'a', 'e', 'i', 'o', 'u'}
print(''.join([c for c in s if c not in vowels]))
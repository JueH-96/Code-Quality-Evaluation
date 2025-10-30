import sys

S = input()

result = ''.join(c for c in S if c not in 'aeiou')
print(result)
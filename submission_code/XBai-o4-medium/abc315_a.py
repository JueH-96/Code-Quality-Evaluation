s = input().strip()
print(''.join(c for c in s if c not in {'a', 'e', 'i', 'o', 'u'}))
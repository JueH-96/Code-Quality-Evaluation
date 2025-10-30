s = input().strip()
parts = s.split('|')
a = [len(part) for part in parts[1:-1]]
print(' '.join(map(str, a)))
s = input().strip()
parts = s.split('|')
a = [str(len(part)) for part in parts[1:-1]]
print(' '.join(a))
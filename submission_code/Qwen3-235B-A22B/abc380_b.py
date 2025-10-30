s = input().strip()
parts = s.split('|')
middle = parts[1:-1]
a = [str(len(part)) for part in middle]
print(' '.join(a))
s = input().strip()
parts = s.split('|')[1:-1]
result = [str(len(part)) for part in parts]
print(' '.join(result))
S = input().strip()
parts = S.split('|')
A = [str(len(part)) for part in parts[1:-1] if part]
print(' '.join(A))
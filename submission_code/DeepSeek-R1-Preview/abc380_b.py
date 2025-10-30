s = input().strip()
parts = s.split('|')
segments = parts[1:-1]
a = [str(len(seg)) for seg in segments]
print(' '.join(a))
s = input().strip()
parts = s.split('|')
segments = parts[1:-1]
a = [len(seg) for seg in segments]
print(' '.join(map(str, a)))
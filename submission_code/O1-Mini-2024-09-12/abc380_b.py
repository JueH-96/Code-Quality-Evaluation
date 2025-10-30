S = input().strip()
segments = S.split('|')
A = [len(s) for s in segments if s]
print(' '.join(map(str, A)))
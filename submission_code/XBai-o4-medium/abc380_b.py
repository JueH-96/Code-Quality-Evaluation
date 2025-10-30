s = input().strip()
parts = s.split('|')
non_empty = [p for p in parts if p]
a = [str(len(p)) for p in non_empty]
print(' '.join(a))
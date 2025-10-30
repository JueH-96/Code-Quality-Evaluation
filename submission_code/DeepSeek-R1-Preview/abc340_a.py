a, b, d = map(int, input().split())
sequence = range(a, b + 1, d)
print(' '.join(map(str, sequence)))
n = int(input())
s = input().strip()
t = input().strip()

def normalize(c):
    if c in {'1', 'l'}:
        return '1'
    elif c in {'0', 'o'}:
        return '0'
    else:
        return c

s_normalized = ''.join([normalize(c) for c in s])
t_normalized = ''.join([normalize(c) for c in t])

print("Yes" if s_normalized == t_normalized else "No")
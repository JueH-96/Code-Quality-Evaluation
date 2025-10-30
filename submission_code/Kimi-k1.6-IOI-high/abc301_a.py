n = int(input())
s = input().strip()

t = s.count('T')
a = s.count('A')

if t > a:
    print('T')
elif a > t:
    print('A')
else:
    required = t
    t_count = 0
    a_count = 0
    for c in s:
        if c == 'T':
            t_count += 1
        else:
            a_count += 1
        if t_count == required:
            print('T')
            break
        if a_count == required:
            print('A')
            break
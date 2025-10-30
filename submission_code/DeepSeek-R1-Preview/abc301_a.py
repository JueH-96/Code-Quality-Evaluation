n = int(input())
s = input().strip()

t_count = s.count('T')
a_count = s.count('A')

if t_count > a_count:
    print('T')
elif a_count > t_count:
    print('A')
else:
    target = t_count  # since t_count == a_count
    t, a = 0, 0
    winner = ''
    for c in s:
        if c == 'T':
            t += 1
        else:
            a += 1
        if t == target:
            winner = 'T'
            break
        if a == target:
            winner = 'A'
            break
    print(winner)
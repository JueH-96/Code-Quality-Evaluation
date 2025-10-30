n = int(input())
s = input().strip()

t_count = s.count('T')
a_count = s.count('A')

if t_count > a_count:
    print('T')
elif a_count > t_count:
    print('A')
else:
    m = t_count  # Since they are equal
    current_t = 0
    current_a = 0
    winner = None
    for c in s:
        if c == 'T':
            current_t += 1
        else:
            current_a += 1
        if current_t == m:
            winner = 'T'
            break
        if current_a == m:
            winner = 'A'
            break
    print(winner)
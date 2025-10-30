n = int(input())
s = input().strip()

t_total = s.count('T')
a_total = s.count('A')

if t_total > a_total:
    print('T')
elif a_total > t_total:
    print('A')
else:
    required = t_total
    current_t = 0
    current_a = 0
    for c in s:
        if c == 'T':
            current_t += 1
        else:
            current_a += 1
        if current_t == required:
            print('T')
            break
        elif current_a == required:
            print('A')
            break
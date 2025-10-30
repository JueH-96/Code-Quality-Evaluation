n = int(input())
s = input()

count_t = s.count('T')
count_a = s.count('A')

if count_t > count_a:
    print('T')
elif count_a > count_t:
    print('A')
else:
    target = count_t
    current_t = 0
    current_a = 0
    winner = ''
    for c in s:
        if c == 'T':
            current_t += 1
        else:
            current_a += 1
        if current_t == target:
            winner = 'T'
            break
        if current_a == target:
            winner = 'A'
            break
    print(winner)
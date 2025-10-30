n = int(input())
s = input().strip()

total_t = s.count('T')
total_a = s.count('A')

if total_t > total_a:
    print('T')
elif total_a > total_t:
    print('A')
else:
    t_step = -1
    a_step = -1
    t_count = 0
    a_count = 0
    for i in range(n):
        if s[i] == 'T':
            t_count += 1
        else:
            a_count += 1
        if t_count == total_t and t_step == -1:
            t_step = i + 1
        if a_count == total_a and a_step == -1:
            a_step = i + 1
    if t_step < a_step:
        print('T')
    else:
        print('A')
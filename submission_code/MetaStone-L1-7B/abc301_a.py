n = int(input())
s = input().strip()

total_t = s.count('T')
total_a = len(s) - total_t

if total_t != total_a:
    print('T' if total_t > total_a else 'A')
else:
    t_index = -1
    a_index = -1
    current_t = 0
    current_a = 0

    for i in range(len(s)):
        if s[i] == 'T':
            current_t += 1
        else:
            current_a += 1

        if current_t == total_t:
            t_index = i
        if current_a == total_a:
            a_index = i

        if t_index != -1 and a_index != -1:
            break

    if t_index < a_index:
        print('T')
    else:
        print('A')
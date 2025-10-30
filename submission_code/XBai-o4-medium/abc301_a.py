n = int(input())
s = input().strip()

count_t = s.count('T')
count_a = s.count('A')

if count_t > count_a:
    print('T')
elif count_a > count_t:
    print('A')
else:
    current_t = 0
    current_a = 0
    k = count_t  # since counts are equal
    for c in s:
        if c == 'T':
            current_t += 1
        else:
            current_a += 1
        if current_t == k:
            print('T')
            break
        if current_a == k:
            print('A')
            break
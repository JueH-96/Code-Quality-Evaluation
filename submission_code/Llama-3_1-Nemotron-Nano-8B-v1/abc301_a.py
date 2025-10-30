n = int(input())
s = input().strip()

count_t = s.count('T')
count_a = s.count('A')

if count_t > count_a:
    print('T')
elif count_a > count_t:
    print('A')
else:
    max_count = count_t
    current_t = 0
    current_a = 0
    step_t = None
    step_a = None
    for i in range(n):
        if s[i] == 'T':
            current_t += 1
        else:
            current_a += 1
        if current_t == max_count and step_t is None:
            step_t = i + 1
        if current_a == max_count and step_a is None:
            step_a = i + 1
        if step_t is not None and step_a is not None:
            break
    print('T' if step_t < step_a else 'A')
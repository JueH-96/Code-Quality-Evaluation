n = int(input())
s = input().strip()

total_t = s.count('T')
total_a = s.count('A')

if total_t > total_a:
    print('T')
elif total_a > total_t:
    print('A')
else:
    # Find the position where T reached total_t
    t_pos = None
    count = 0
    for i in range(len(s)):
        if s[i] == 'T':
            count += 1
            if count == total_t:
                t_pos = i
                break
    
    # Find the position where A reached total_a
    a_pos = None
    count = 0
    for i in range(len(s)):
        if s[i] == 'A':
            count += 1
            if count == total_a:
                a_pos = i
                break
    
    if t_pos < a_pos:
        print('T')
    else:
        print('A')
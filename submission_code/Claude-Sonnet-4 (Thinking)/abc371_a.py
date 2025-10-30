s_ab, s_ac, s_bc = input().split()

a_older_count = 0
b_older_count = 0  
c_older_count = 0

if s_ab == '>':
    a_older_count += 1
else:
    b_older_count += 1

if s_ac == '>':
    a_older_count += 1
else:
    c_older_count += 1

if s_bc == '>':
    b_older_count += 1
else:
    c_older_count += 1

if a_older_count == 1:
    print('A')
elif b_older_count == 1:
    print('B')
else:
    print('C')
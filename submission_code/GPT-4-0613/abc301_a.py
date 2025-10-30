N = int(input())
S = input()

T_count = 0
A_count = 0

for i in S:
    if i == 'T':
        T_count += 1
    else:
        A_count += 1
    if T_count > A_count:
        print('T')
        break
    elif A_count > T_count:
        print('A')
        break
else:
    if T_count > A_count:
        print('T')
    elif A_count > T_count:
        print('A')
    else:
        print('T')
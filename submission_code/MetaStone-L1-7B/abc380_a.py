s = input().strip()
count_1 = s.count('1')
count_2 = s.count('2')
count_3 = s.count('3')

if count_1 == 1 and count_2 == 2 and count_3 == 3:
    print("Yes")
else:
    print("No")
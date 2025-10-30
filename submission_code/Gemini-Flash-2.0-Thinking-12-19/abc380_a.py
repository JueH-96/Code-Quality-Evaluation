n_str = input()

count1 = 0
count2 = 0
count3 = 0

for digit in n_str:
    if digit == '1':
        count1 += 1
    elif digit == '2':
        count2 += 1
    elif digit == '3':
        count3 += 1

if count1 == 1 and count2 == 2 and count3 == 3:
    print("Yes")
else:
    print("No")
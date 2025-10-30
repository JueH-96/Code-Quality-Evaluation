n = input().strip()
count1 = n.count('1')
count2 = n.count('2')
count3 = n.count('3')
if count1 == 1 and count2 == 2 and count3 == 3:
    print("Yes")
else:
    print("No")
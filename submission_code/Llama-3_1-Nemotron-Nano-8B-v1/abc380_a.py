s = input().strip()
count1 = s.count('1')
count2 = s.count('2')
count3 = s.count('3')
if count1 == 1 and count2 == 2 and count3 == 3 and (count1 + count2 + count3 == 6):
    print("Yes")
else:
    print("No")
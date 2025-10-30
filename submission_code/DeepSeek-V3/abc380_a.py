# YOUR CODE HERE
N = input().strip()
count1 = N.count('1')
count2 = N.count('2')
count3 = N.count('3')
if count1 == 1 and count2 == 2 and count3 == 3:
    print("Yes")
else:
    print("No")
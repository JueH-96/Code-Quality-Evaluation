# YOUR CODE HERE
y = int(input())
if y % 400 == 0:
    print(366)
elif y % 100 == 0:
    print(365)
elif y % 4 == 0:
    print(366)
else:
    print(365)
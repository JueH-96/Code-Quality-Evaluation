# YOUR CODE HERE
R = int(input())

if R < 100:
    # Currently 1 ^, next level at 100
    print(100 - R)
elif R < 200:
    # Currently 2 ^, next level at 200
    print(200 - R)
else:
    # Currently 3 ^, next level at 300
    print(300 - R)
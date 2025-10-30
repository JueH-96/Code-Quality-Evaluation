# YOUR CODE HERE
R = int(input())
if R < 100:
    required = 100 - R
elif R < 200:
    required = 200 - R
elif R < 300:
    required = 300 - R
else:
    required = 0
print(required)
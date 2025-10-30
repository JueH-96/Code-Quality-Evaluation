# YOUR CODE HERE
rating = int(input())
if rating <= 99:
    print(100 - rating)
elif rating <= 199:
    print(200 - rating)
else:
    print(300 - rating)
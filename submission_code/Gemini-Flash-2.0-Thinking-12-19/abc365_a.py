# YOUR CODE HERE
year = int(input())

if year % 400 == 0:
    days = 366
elif year % 100 == 0:
    days = 365
elif year % 4 == 0:
    days = 366
else:
    days = 365

print(days)
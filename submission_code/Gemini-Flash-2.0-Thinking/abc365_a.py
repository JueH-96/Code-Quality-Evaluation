y = int(input())

if y % 400 == 0:
    days = 366
elif y % 100 == 0:
    days = 365
elif y % 4 == 0:
    days = 366
else:
    days = 365

print(days)
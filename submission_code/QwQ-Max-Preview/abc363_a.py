R = int(input())
if R < 100:
    target = 100
elif R < 200:
    target = 200
else:
    target = 300
print(target - R)
# Read the input
R = int(input())

# Determine the minimum increase needed
if R < 100:
    print(100 - R)
elif R < 200:
    print(200 - R)
else:
    print(300 - R)
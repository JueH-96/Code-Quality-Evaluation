# YOUR CODE HERE
a, b = map(int, input().split())

count = 0
if (2 * a - b) % 1 == 0 and 2 * a - b != 0:
    count += 1
if (a + b) % 2 == 0 :
    count += 1
if (2 * b - a) % 1 == 0 and 2 * b - a != 0:
    count +=1

if a == b:
    if (2 * a - b) % 1 == 0 and 2 * a - b != 0:
        count = 1
    elif (a + b) % 2 == 0:
        count = 1
    elif (2 * b - a) % 1 == 0 and 2 * b - a != 0:
        count = 1
    else:
        count = 1
elif 2 * a - b == 0 and 2 * b - a == 0:
    count = 1
elif 2 * a - b == 0 or 2 * b - a == 0 or (a + b) % 2 == 0:
    if 2 * a - b == 0 and (a + b) % 2 == 0:
        count = 2
    elif 2 * b - a == 0 and (a + b) % 2 == 0:
        count = 2
    elif 2 * a - b == 0 and 2 * b - a == 0:
        count = 1

print(count)
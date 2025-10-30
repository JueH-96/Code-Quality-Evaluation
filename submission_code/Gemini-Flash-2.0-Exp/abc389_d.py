R = int(input())
count = 0
for i in range(-R, R + 1):
    for j in range(-R, R + 1):
        if (i + 0.5)**2 + (j + 0.5)**2 <= R**2 and \
           (i + 0.5)**2 + (j - 0.5)**2 <= R**2 and \
           (i - 0.5)**2 + (j + 0.5)**2 <= R**2 and \
           (i - 0.5)**2 + (j - 0.5)**2 <= R**2:
            count += 1
print(count)
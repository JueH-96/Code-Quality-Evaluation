# YOUR CODE HERE
N = int(input())
count = 0
for b in range(2, 61):
    a = 1
    while True:
        x = a**b
        if x > N:
            break
        if x > 0:
            count += 1
        a += 1
print(count)
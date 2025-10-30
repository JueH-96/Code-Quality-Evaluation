# YOUR CODE HERE
N = int(input())
count = 0
while N % 2 == 0:
    count += 1
    N //= 2
print(count)
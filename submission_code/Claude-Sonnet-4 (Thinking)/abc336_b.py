# YOUR CODE HERE
N = int(input())
count = 0
while N % 2 == 0:
    N //= 2
    count += 1
print(count)
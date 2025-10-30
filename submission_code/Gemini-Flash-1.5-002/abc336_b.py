# YOUR CODE HERE
n = int(input())
count = 0
while n > 0 and n % 2 == 0:
    count += 1
    n //= 2
print(count)
# YOUR CODE HERE
import math

def count_divisors(n):
    count = 0
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            if i * i == n:
                count += 1
            else:
                count += 2
    return count

def count_numbers_with_9_divisors(N):
    count = 0
    for num in range(1, int(math.sqrt(N)) + 1):
        square = num * num
        if square > N:
            break
        if count_divisors(square) == 9:
            count += 1
    return count

N = int(input())
result = count_numbers_with_9_divisors(N)
print(result)
# YOUR CODE HERE
def is_326_like(num):
    hundreds = num // 100
    tens = (num % 100) // 10
    ones = num % 10
    return hundreds * tens == ones

N = int(input())

while not is_326_like(N):
    N += 1

print(N)
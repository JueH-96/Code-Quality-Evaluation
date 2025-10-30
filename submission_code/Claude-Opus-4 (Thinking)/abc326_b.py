# YOUR CODE HERE
N = int(input())

def is_326_like(num):
    hundreds = num // 100
    tens = (num // 10) % 10
    ones = num % 10
    return hundreds * tens == ones

while not is_326_like(N):
    N += 1

print(N)
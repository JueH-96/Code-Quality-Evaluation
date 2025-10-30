# YOUR CODE HERE
n = int(input())

def is_326_like(num):
    hundreds = num // 100
    tens = (num % 100) // 10
    ones = num % 10
    return hundreds * tens == ones

found = False
i = n
while not found:
    if is_326_like(i):
        print(i)
        found = True
    i += 1
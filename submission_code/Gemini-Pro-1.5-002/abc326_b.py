# YOUR CODE HERE
n = int(input())
for i in range(n, 1000):
    hundreds = i // 100
    tens = (i % 100) // 10
    ones = i % 10
    if hundreds * tens == ones:
        print(i)
        break
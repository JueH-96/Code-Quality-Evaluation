N = int(input())

for number in range(N, 1000):
    hundreds = number // 100
    tens = (number % 100) // 10
    ones = number % 10
    if hundreds * tens == ones:
        print(number)
        break
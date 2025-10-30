# YOUR CODE HERE

def is_power_of_two_and_three(n):
    x = 0
    while n % 2 == 0:
        n /= 2
        x += 1
    y = 0
    while n % 3 == 0:
        n /= 3
        y += 1
    return (n == 1) and (x > 0 and y > 0)

n = int(input())
print('Yes' if is_power_of_two_and_three(n) else 'No')
# YOUR CODE HERE

def is_power_of_two(n):
    return (n & (n - 1)) == 0

def is_power_of_three(n):
    while n % 3 == 0:
        n /= 3
    return n == 1

def solve(n):
    if n == 1:
        return "Yes"
    elif n % 2 == 0 and is_power_of_two(n // 2):
        return "Yes"
    elif n % 3 == 0 and is_power_of_three(n // 3):
        return "Yes"
    else:
        return "No"

n = int(input())
print(solve(n))
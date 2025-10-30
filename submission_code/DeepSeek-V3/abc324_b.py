# YOUR CODE HERE
def is_power_of_two(n):
    return (n != 0) and (n & (n - 1)) == 0

def is_power_of_three(n):
    if n < 1:
        return False
    while n % 3 == 0:
        n = n // 3
    return n == 1

def can_be_expressed(N):
    if N == 1:
        return True
    x = 0
    while N % 2 == 0:
        N = N // 2
        x += 1
    y = 0
    while N % 3 == 0:
        N = N // 3
        y += 1
    if N == 1:
        return True
    else:
        return False

N = int(input())
if can_be_expressed(N):
    print("Yes")
else:
    print("No")
import sys

def is_power_of_2_and_3(n):
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    return n == 1

n = int(input())
print("Yes" if is_power_of_2_and_3(n) else "No")
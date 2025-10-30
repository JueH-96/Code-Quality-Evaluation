import math

def check_powers_of_2_and_3(n):
    while n % 2 == 0:
        n = n // 2
    while n % 3 == 0:
        n = n // 3
    return n == 1

n = int(input())
if check_powers_of_2_and_3(n):
    print("Yes")
else:
    print("No")
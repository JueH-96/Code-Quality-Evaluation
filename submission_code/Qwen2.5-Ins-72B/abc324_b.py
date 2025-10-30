# YOUR CODE HERE
import math

def is_power_of_2_and_3(N):
    if N == 1:
        return True
    while N % 2 == 0:
        N //= 2
    while N % 3 == 0:
        N //= 3
    return N == 1

N = int(input())
if is_power_of_2_and_3(N):
    print("Yes")
else:
    print("No")
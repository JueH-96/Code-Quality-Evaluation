# YOUR CODE HERE
import sys
input = sys.stdin.read

def total_payment(N):
    total_cost = 0
    while N >= 2:
        total_cost += N
        N = N // 2
    return total_cost

N = int(input().strip())
print(total_payment(N))
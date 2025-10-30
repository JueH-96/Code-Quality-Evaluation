# YOUR CODE HERE
import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    MOD = 998244353
    total_sum = 0
    cumulative_sum = 0
    power_of_10 = 1
    
    for i in range(N-1, -1, -1):
        a = A[i]
        while a >= power_of_10:
            power_of_10 *= 10
        total_sum = (total_sum + (cumulative_sum + a * power_of_10) % MOD) % MOD
        cumulative_sum = (cumulative_sum + a) % MOD
    
    print(total_sum)

solve()
# YOUR CODE HERE

import sys

def solve():
    MOD = 998244353
    N, X = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))

    # Calculate the total time
    total_time = sum(T)

    # Calculate the probability of the first song being played
    prob = (X + T[0] / 2) / total_time

    # Convert the probability to an integer
    prob = int(prob * MOD)

    print(prob)

solve()
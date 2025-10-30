# YOUR CODE HERE
import sys

def solve():
    N = int(sys.stdin.readline())

    # Repeatedly divide N by 2 until it's no longer divisible by 2
    # This removes all factors of 2 from N
    while N % 2 == 0:
        N //= 2

    # Repeatedly divide N by 3 until it's no longer divisible by 3
    # This removes all factors of 3 from the remaining N
    while N % 3 == 0:
        N //= 3

    # If N has been reduced to 1, it means the original N was
    # composed solely of prime factors 2 and 3.
    if N == 1:
        print("Yes")
    else:
        # Otherwise, N still contains a prime factor other than 2 or 3.
        print("No")

if __name__ == '__main__':
    solve()
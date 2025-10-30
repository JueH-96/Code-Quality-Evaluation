# YOUR CODE HERE
import sys

def is_power_of_two_and_three(n):
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    return n == 1

def main():
    N = int(sys.stdin.read().strip())
    if is_power_of_two_and_three(N):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
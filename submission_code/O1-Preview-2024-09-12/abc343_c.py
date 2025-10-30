# YOUR CODE HERE
import sys
import math

def main():
    N = int(sys.stdin.readline())
    x_max = int(N ** (1/3)) + 1
    for x in range(x_max, 0, -1):
        K = x ** 3
        if K > N:
            continue
        if str(K) == str(K)[::-1]:
            print(K)
            return

if __name__ == "__main__":
    main()
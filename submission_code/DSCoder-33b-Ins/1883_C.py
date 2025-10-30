# YOUR CODE HERE

import sys

def min_operations(a, k):
    product = 1
    for i in a:
        product *= i
    if product % k == 0:
        return 0
    else:
        return 1

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n, k = map(int, sys.stdin.readline().split())
        a = list(map(int, sys.stdin.readline().split()))
        print(min_operations(a, k))

if __name__ == "__main__":
    main()
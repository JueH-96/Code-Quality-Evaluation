import sys
import itertools
from functools import reduce

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    max_xor = 0
    for combo in itertools.combinations(A, K):
        current = reduce(lambda x, y: x ^ y, combo, 0)
        if current > max_xor:
            max_xor = current
    print(max_xor)

if __name__ == "__main__":
    main()
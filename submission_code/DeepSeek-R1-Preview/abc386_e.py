import sys
from itertools import combinations
from functools import reduce

def main():
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    max_xor = 0
    for combo in combinations(a, k):
        current = reduce(lambda x, y: x ^ y, combo, 0)
        if current > max_xor:
            max_xor = current
    print(max_xor)

if __name__ == "__main__":
    main()
import itertools
import math
import sys

def is_square(n):
    return int(math.isqrt(n)) ** 2 == n

def count_square_permutations(N, S):
    # Generate all permutations of the string S
    permutations = set(itertools.permutations(S))
    count = 0
    unique_squares = set()

    for perm in permutations:
        num = int(''.join(perm))
        if is_square(num):
            unique_squares.add(num)

    return len(unique_squares)

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = data[1]

    result = count_square_permutations(N, S)
    print(result)

if __name__ == "__main__":
    main()
import math
from itertools import permutations

def is_perfect_square(num):
    sqrt_num = math.isqrt(num)
    return sqrt_num * sqrt_num == num

def count_square_permutations(S):
    unique_squares = set()
    
    for perm in permutations(S):
        num = int(''.join(perm))
        if is_perfect_square(num):
            unique_squares.add(num)
    
    return len(unique_squares)

def main():
    N = int(input().strip())
    S = input().strip()
    
    result = count_square_permutations(S)
    print(result)

if __name__ == "__main__":
    main()
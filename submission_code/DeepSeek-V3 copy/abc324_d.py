import sys
import math
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    
    # Count the frequency of each digit in S
    digit_counts = defaultdict(int)
    for c in S:
        digit_counts[int(c)] += 1
    
    # Generate all possible square numbers with the same number of digits as S
    min_num = 10 ** (N-1)
    max_num = 10 ** N - 1
    min_square_root = math.isqrt(min_num)
    max_square_root = math.isqrt(max_num)
    
    square_numbers = set()
    for i in range(min_square_root, max_square_root + 1):
        square = i * i
        square_numbers.add(square)
    
    # For each square number, check if it can be formed by the digits of S
    count = 0
    for square in square_numbers:
        # Convert the square number to a string and pad with leading zeros if necessary
        square_str = str(square).zfill(N)
        # Count the frequency of each digit in the square number
        square_digit_counts = defaultdict(int)
        for c in square_str:
            square_digit_counts[int(c)] += 1
        # Check if the digit counts match
        if square_digit_counts == digit_counts:
            count += 1
    
    print(count)

if __name__ == "__main__":
    main()
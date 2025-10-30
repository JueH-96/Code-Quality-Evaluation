import sys
import math
from collections import defaultdict

def is_square(n):
    return int(math.isqrt(n))**2 == n

def count_square_pairs(N, A):
    count = 0
    # Dictionary to store the count of each number
    num_count = defaultdict(int)

    for i in range(N):
        for j in range(i+1, N):
            if is_square(A[i] * A[j]):
                count += 1

    return count

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    result = count_square_pairs(N, A)
    print(result)

if __name__ == "__main__":
    main()
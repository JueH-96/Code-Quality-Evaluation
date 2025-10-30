import math
from collections import defaultdict

def is_square(n):
    return int(math.isqrt(n)) ** 2 == n

def count_square_pairs(N, A):
    count = 0
    freq = defaultdict(int)
    
    for a in A:
        for k, v in freq.items():
            if is_square(a * k):
                count += v
        freq[a] += 1
    
    return count

if __name__ == "__main__":
    N = int(input())
    A = list(map(int, input().split()))
    print(count_square_pairs(N, A))
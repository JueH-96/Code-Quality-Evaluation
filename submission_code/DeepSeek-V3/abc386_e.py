# YOUR CODE HERE
import sys
from itertools import combinations

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    max_xor = 0
    for comb in combinations(A, K):
        current_xor = 0
        for num in comb:
            current_xor ^= num
        if current_xor > max_xor:
            max_xor = current_xor
    print(max_xor)

if __name__ == "__main__":
    main()
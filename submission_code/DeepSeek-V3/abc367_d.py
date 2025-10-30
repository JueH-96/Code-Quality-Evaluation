# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Compute the prefix sum modulo M
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = (prefix[i] + A[i]) % M
    
    # Count the frequency of each prefix sum
    freq = defaultdict(int)
    for p in prefix:
        freq[p] += 1
    
    # Calculate the number of pairs
    result = 0
    for p in freq:
        cnt = freq[p]
        result += cnt * (cnt - 1) // 2
    
    print(result)

if __name__ == "__main__":
    main()
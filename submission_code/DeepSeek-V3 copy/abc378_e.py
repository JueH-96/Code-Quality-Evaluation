# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = (prefix[i] + A[i]) % M
    
    count = defaultdict(int)
    result = 0
    for p in prefix:
        result += count[p]
        count[p] += 1
    
    print(result)

if __name__ == "__main__":
    main()
import sys
from collections import defaultdict

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Count the occurrences of each number
    count = defaultdict(int)
    for a in A:
        count[a] += 1
    
    # Calculate prefix sums for each number
    prefix_sum = defaultdict(int)
    prefix_sums = [defaultdict(int) for _ in range(N+1)]
    for i, a in enumerate(A):
        prefix_sum[a] += 1
        for k, v in prefix_sum.items():
            prefix_sums[i+1][k] = v
    
    # Calculate the number of valid triples
    result = 0
    for i, a in enumerate(A):
        for j in range(i+1, N):
            if A[j] != a:
                result += count[a] - prefix_sums[j+1][a]
    
    print(result)

if __name__ == "__main__":
    solve()
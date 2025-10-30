# YOUR CODE HERE
import sys
from collections import defaultdict

def count_triples(N, A):
    count = 0
    prefix_count = defaultdict(int)
    suffix_count = defaultdict(int)
    
    for num in A:
        suffix_count[num] += 1
    
    for j in range(1, N - 1):
        suffix_count[A[j]] -= 1
        if A[j] in prefix_count:
            for num in prefix_count:
                if num != A[j] and num in suffix_count and suffix_count[num] > 0:
                    count += prefix_count[num] * suffix_count[num]
        prefix_count[A[j]] += 1
    
    return count

N = int(input())
A = list(map(int, input().split()))

print(count_triples(N, A))
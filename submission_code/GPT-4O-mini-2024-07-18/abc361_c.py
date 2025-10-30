def min_max_difference(N, K, A):
    A.sort()
    # We need to remove K elements, which means we will keep N - K elements
    # The best way to minimize the max - min of the remaining elements is to
    # remove elements from the ends of the sorted array.
    
    # The minimum possible value of max(B) - min(B) can be found by considering
    # the subarray of length N - K in the sorted array.
    
    min_diff = float('inf')
    
    for i in range(K + 1):
        # We remove the first i elements and the last K - i elements
        current_min = A[i]
        current_max = A[N - (K - i) - 1]
        min_diff = min(min_diff, current_max - current_min)
    
    return min_diff

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

result = min_max_difference(N, K, A)
print(result)
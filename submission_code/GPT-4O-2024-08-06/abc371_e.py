# YOUR CODE HERE
def calculate_distinct_subarray_sum(N, A):
    freq = {}
    distinct_count = 0
    total_sum = 0
    j = 0
    
    for i in range(N):
        while j < N:
            if A[j] in freq:
                freq[A[j]] += 1
            else:
                freq[A[j]] = 1
                distinct_count += 1
            
            total_sum += distinct_count
            j += 1
        
        # Move i to the right, adjust the frequency and distinct count
        freq[A[i]] -= 1
        if freq[A[i]] == 0:
            del freq[A[i]]
            distinct_count -= 1
    
    return total_sum

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))

result = calculate_distinct_subarray_sum(N, A)
print(result)
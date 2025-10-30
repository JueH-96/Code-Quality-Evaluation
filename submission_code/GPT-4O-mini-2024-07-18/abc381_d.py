def max_1122_sequence_length(N, A):
    max_length = 0
    
    for start in range(N):
        count = {}
        valid = True
        for end in range(start, N):
            num = A[end]
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
            
            # Check if the current subarray can still be a 1122 sequence
            if count[num] > 2:
                valid = False
                break
            
            # Check if the length of the current subarray is even
            current_length = end - start + 1
            if current_length % 2 == 0:
                # Check if all numbers appear exactly twice
                if all(v == 2 for v in count.values() if v > 0):
                    max_length = max(max_length, current_length)
    
    return max_length

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))

result = max_1122_sequence_length(N, A)
print(result)
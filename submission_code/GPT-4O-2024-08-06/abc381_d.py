# YOUR CODE HERE
def find_max_1122_sequence_length(N, A):
    from collections import defaultdict
    
    max_length = 0
    start = 0
    count = defaultdict(int)
    
    for end in range(N):
        count[A[end]] += 1
        
        # Check if the current window is valid
        while count[A[end]] > 2:
            count[A[start]] -= 1
            start += 1
        
        # Check if the current window is a valid 1122 sequence
        if (end - start + 1) % 2 == 0:
            valid = True
            for value in count.values():
                if value != 0 and value != 2:
                    valid = False
                    break
            if valid:
                max_length = max(max_length, end - start + 1)
    
    print(max_length)

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

find_max_1122_sequence_length(N, A)
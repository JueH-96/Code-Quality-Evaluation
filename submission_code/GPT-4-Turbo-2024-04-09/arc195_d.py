import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        A = list(map(int, data[index:index+N]))
        index += N
        
        # To minimize operations, we need to find the longest sequences of equal elements
        # and the minimum swaps needed to make such sequences contiguous if they are not.
        
        # Count the number of operations needed
        operations = 0
        i = 0
        while i < N:
            # Start of a block
            start = i
            while i + 1 < N and A[i] == A[i + 1]:
                i += 1
            # i is now at the last element of the current block of equal elements
            operations += 1
            i += 1  # move to the next new element or out of bounds
        
        results.append(str(operations))
    
    # Print all results for each test case
    print("
".join(results))
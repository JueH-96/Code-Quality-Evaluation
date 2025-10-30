def is_1122_sequence(arr):
    # Check if length is even
    if len(arr) % 2 != 0:
        return False
    
    # Check if adjacent pairs are equal
    for i in range(0, len(arr), 2):
        if arr[i] != arr[i+1]:
            return False
    
    # Check if each number appears exactly twice
    count = {}
    for num in arr:
        count[num] = count.get(num, 0) + 1
    
    for val in count.values():
        if val != 2:
            return False
            
    return True

def solve(N, A):
    max_length = 0
    
    # Try all possible subarrays of even length
    for i in range(N):
        for j in range(i+1, N+1):
            # Only check subarrays of even length
            if (j - i) % 2 == 0:
                subarray = A[i:j]
                if is_1122_sequence(subarray):
                    max_length = max(max_length, j - i)
    
    return max_length

# Read input
N = int(input())
A = list(map(int, input().split()))

# Print result
print(solve(N, A))
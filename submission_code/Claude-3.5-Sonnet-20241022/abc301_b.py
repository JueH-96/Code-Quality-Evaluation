N = int(input())
A = list(map(int, input().split()))

def find_first_non_consecutive(arr):
    for i in range(len(arr)-1):
        if abs(arr[i] - arr[i+1]) != 1:
            return i
    return -1

while True:
    idx = find_first_non_consecutive(A)
    if idx == -1:
        break
        
    curr = A[idx]
    next_val = A[idx+1]
    
    if curr < next_val:
        # Insert ascending sequence
        to_insert = list(range(curr+1, next_val))
        A[idx+1:idx+1] = to_insert
    else:
        # Insert descending sequence
        to_insert = list(range(curr-1, next_val, -1))
        A[idx+1:idx+1] = to_insert

print(*A)
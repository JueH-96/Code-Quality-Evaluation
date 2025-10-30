n = int(input())
A = list(map(int, input().split()))

while True:
    # Check if all adjacent pairs have absolute difference 1
    all_diff_one = True
    for i in range(len(A) - 1):
        if abs(A[i] - A[i+1]) != 1:
            all_diff_one = False
            break
    
    if all_diff_one:
        break
    
    # Find the first pair whose absolute difference is not 1
    for i in range(len(A) - 1):
        if abs(A[i] - A[i+1]) != 1:
            # Insert numbers between A[i] and A[i+1]
            if A[i] < A[i+1]:
                # Insert A[i]+1, A[i]+2, ..., A[i+1]-1
                to_insert = list(range(A[i] + 1, A[i+1]))
            else:
                # Insert A[i]-1, A[i]-2, ..., A[i+1]+1
                to_insert = list(range(A[i] - 1, A[i+1], -1))
            
            # Insert these numbers between A[i] and A[i+1]
            A = A[:i+1] + to_insert + A[i+1:]
            break

print(' '.join(map(str, A)))
n = int(input())
h = list(map(int, input().split()))

a = [0] * (n + 1)  # A_0, A_1, ..., A_N
answer = [0] * n
found = [False] * n
operation = 0

while not all(found):
    operation += 1
    
    # Step 1: Increase A_0 by 1
    a[0] += 1
    
    # Step 2: For i=1,2,...,N
    for i in range(1, n + 1):
        if a[i-1] > a[i] and a[i-1] > h[i-1]:
            a[i-1] -= 1
            a[i] += 1
    
    # Check if any A_i became positive for the first time
    for i in range(1, n + 1):
        if not found[i-1] and a[i] > 0:
            answer[i-1] = operation
            found[i-1] = True

print(' '.join(map(str, answer)))
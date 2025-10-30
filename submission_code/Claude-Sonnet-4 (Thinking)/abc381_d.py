N = int(input())
A = list(map(int, input().split()))

max_length = 0

for i in range(N - 1):
    if A[i] != A[i + 1]:
        continue
    
    seen = set()
    length = 0
    pos = i
    
    while pos < N - 1:
        if A[pos] != A[pos + 1]:
            break
        
        # We have a pair A[pos], A[pos+1]
        num = A[pos]
        if num in seen:
            # This number already appeared twice
            break
        
        seen.add(num)
        length += 2
        pos += 2
    
    max_length = max(max_length, length)

print(max_length)
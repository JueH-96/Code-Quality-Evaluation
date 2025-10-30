# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

while True:
    # Find the first pair of adjacent terms with absolute difference > 1
    found = False
    for i in range(len(a) - 1):
        if abs(a[i] - a[i + 1]) > 1:
            # Insert numbers between a[i] and a[i+1]
            if a[i] < a[i + 1]:
                # Insert a[i]+1, a[i]+2, ..., a[i+1]-1
                to_insert = list(range(a[i] + 1, a[i + 1]))
            else:
                # Insert a[i]-1, a[i]-2, ..., a[i+1]+1
                to_insert = list(range(a[i] - 1, a[i + 1], -1))
            
            # Insert the numbers at position i+1
            a = a[:i + 1] + to_insert + a[i + 1:]
            found = True
            break
    
    if not found:
        break

print(' '.join(map(str, a)))
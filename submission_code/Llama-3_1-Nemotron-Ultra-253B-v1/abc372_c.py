import sys

n, q = map(int, sys.stdin.readline().split())
s = list(sys.stdin.readline().strip())
count = 0

for i in range(n - 2):
    if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
        count += 1

for _ in range(q):
    x, c = sys.stdin.readline().split()
    x = int(x) - 1  # convert to 0-based index
    start = max(0, x - 2)
    end = min(x, n - 3)
    
    # Subtract existing ABCs in the affected range
    for i in range(start, end + 1):
        if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
            count -= 1
    
    # Update the character
    s[x] = c
    
    # Add new ABCs in the affected range
    for i in range(start, end + 1):
        if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
            count += 1
    
    print(count)
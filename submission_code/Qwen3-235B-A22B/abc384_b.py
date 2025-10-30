# Read input
n, r = map(int, input().split())
current = r

for _ in range(n):
    d, a = map(int, input().split())
    if d == 1:
        # Check Div 1 range
        if 1600 <= current <= 2799:
            current += a
    else:
        # Check Div 2 range
        if 1200 <= current <= 2399:
            current += a

# Output the final rating
print(current)
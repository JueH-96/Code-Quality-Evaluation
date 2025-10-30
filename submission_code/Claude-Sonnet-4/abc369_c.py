n = int(input())
a = list(map(int, input().split()))

count = 0

for i in range(n):
    count += 1  # Single element at position i
    
    if i < n - 1:
        diff = a[i + 1] - a[i]
        j = i + 1
        
        # Extend the AP as far as possible
        while j < n:
            if j == i + 1 or a[j] - a[j - 1] == diff:
                if j > i:
                    count += 1  # Pair (i, j)
                j += 1
            else:
                break

print(count)
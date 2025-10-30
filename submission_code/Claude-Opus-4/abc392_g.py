# YOUR CODE HERE
n = int(input())
s = list(map(int, input().split()))

# Convert to set for O(1) lookup
s_set = set(s)

count = 0

# Check all pairs (A, C) where A < C
for i in range(n):
    for j in range(i + 1, n):
        a = s[i]
        c = s[j]
        
        # Ensure a < c
        if a > c:
            a, c = c, a
        
        # Check if (a + c) is even
        if (a + c) % 2 == 0:
            b = (a + c) // 2
            
            # Check if b is in the set and a < b < c
            if b in s_set and a < b < c:
                count += 1

print(count)
n, q = map(int, input().split())
s = list(input().strip())
count = 0

# Calculate initial count of ABC substrings
for i in range(n - 2):
    if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
        count += 1

ans = []
for _ in range(q):
    x, c = input().split()
    x = int(x) - 1  # Convert to 0-based index
    
    # Check and subtract old ABC substrings affected by the change
    for start in [x-2, x-1, x]:
        if 0 <= start <= n - 3:
            if s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C':
                count -= 1
    
    # Update the character
    s[x] = c
    
    # Check and add new ABC substrings affected by the change
    for start in [x-2, x-1, x]:
        if 0 <= start <= n - 3:
            if s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C':
                count += 1
    
    ans.append(count)

for num in ans:
    print(num)
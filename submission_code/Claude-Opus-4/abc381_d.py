# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

max_length = 0

# Try each starting position
for start in range(n):
    # For each starting position, try different lengths
    for end in range(start, n + 1):
        length = end - start
        
        # Skip if length is odd
        if length % 2 == 1:
            continue
            
        # Check if this subarray is a 1122 sequence
        is_valid = True
        
        # Check condition 2: pairs must be equal
        for i in range(0, length, 2):
            if start + i + 1 < end and a[start + i] != a[start + i + 1]:
                is_valid = False
                break
        
        if not is_valid:
            continue
            
        # Check condition 3: each number appears 0 or 2 times
        count = {}
        for i in range(start, end):
            count[a[i]] = count.get(a[i], 0) + 1
        
        for c in count.values():
            if c != 2:
                is_valid = False
                break
        
        if is_valid:
            max_length = max(max_length, length)

print(max_length)
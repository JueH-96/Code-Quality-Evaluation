# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

max_len = 0

for start in range(n):
    count = {}
    
    i = start
    while i + 1 < n:
        # Current pair must be equal
        if a[i] != a[i + 1]:
            break
        
        # Add pair to count
        num = a[i]
        count[num] = count.get(num, 0) + 2
        
        # If any number appears more than 2 times, stop
        if count[num] > 2:
            break
        
        i += 2
        
        # Check if current sequence is valid
        if all(c == 2 for c in count.values()):
            max_len = max(max_len, i - start)

print(max_len)
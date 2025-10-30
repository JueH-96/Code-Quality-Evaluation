# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

total = 0
for L in range(n):
    present = [False] * (n + 2)  # present[i] = True if value i is in A[L:R+1]
    segments = 0
    for R in range(L, n):
        val = a[R]
        if not present[val]:
            present[val] = True
            # Check if this value connects to existing segments
            left_connected = present[val - 1]
            right_connected = present[val + 1]
            
            if left_connected and right_connected:
                # Bridges two segments
                segments -= 1
            elif not left_connected and not right_connected:
                # New isolated segment
                segments += 1
            # else: extends one segment, no change in count
        
        total += segments

print(total)
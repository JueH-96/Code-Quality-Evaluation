n, m = map(int, input().split())
B = list(map(int, input().split()))
W = list(map(int, input().split()))

# Sort in descending order to pick best balls first
B.sort(reverse=True)
W.sort(reverse=True)

max_sum = 0

# Try each possible number of white balls
for num_white in range(m + 1):
    # Sum of the best num_white white balls
    white_sum = sum(W[:num_white])
    
    # We need at least num_white black balls
    # Try different numbers of black balls starting from num_white
    black_sum = 0
    for num_black in range(num_white, n + 1):
        # Add the next best black ball
        if num_black > 0:
            black_sum += B[num_black - 1]
        
        total_sum = white_sum + black_sum
        max_sum = max(max_sum, total_sum)

print(max_sum)
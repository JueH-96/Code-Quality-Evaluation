# YOUR CODE HERE
N, M = map(int, input().split())
B = list(map(int, input().split()))
W = list(map(int, input().split()))

# Sort in descending order
B.sort(reverse=True)
W.sort(reverse=True)

# Compute prefix sums for efficiency
B_prefix = [0]
for b in B:
    B_prefix.append(B_prefix[-1] + b)

W_prefix = [0]
for w in W:
    W_prefix.append(W_prefix[-1] + w)

max_sum = 0

# Try all possible numbers of white balls
for num_white in range(M + 1):
    # Sum of chosen white balls
    white_sum = W_prefix[num_white]
    
    # We need at least num_white black balls
    # But we can choose more if they have positive values
    black_sum = 0
    
    # Choose all positive black balls and enough negative ones if needed
    positive_count = 0
    for i in range(N):
        if B[i] > 0:
            positive_count += 1
        else:
            break
    
    if positive_count >= num_white:
        # We have enough positive black balls
        black_sum = B_prefix[positive_count]
    else:
        # We need some negative black balls too
        black_sum = B_prefix[num_white]
    
    # We can always choose more black balls if they're positive
    for i in range(num_white, N):
        if B[i] > 0:
            black_sum += B[i]
    
    total_sum = white_sum + black_sum
    max_sum = max(max_sum, total_sum)

print(max_sum)
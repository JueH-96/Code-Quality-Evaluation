# YOUR CODE HERE
N, K = map(int, input().split())
S = input().strip()

count = 0
i = 0

while i <= N - K:
    # Check if we have K consecutive healthy teeth starting at position i
    can_eat = True
    for j in range(i, i + K):
        if S[j] != 'O':
            can_eat = False
            break
    
    if can_eat:
        # Eat the strawberry using these K teeth
        count += 1
        # Skip past these K teeth since they now have cavities
        i += K
    else:
        # Move to the next position
        i += 1

print(count)
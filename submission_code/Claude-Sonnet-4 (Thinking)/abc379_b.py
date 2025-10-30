N, K = map(int, input().split())
S = list(input().strip())

count = 0
i = 0

while i <= N - K:
    # Check if we have K consecutive healthy teeth starting at position i
    all_healthy = True
    for j in range(i, i + K):
        if S[j] != 'O':
            all_healthy = False
            break
    
    if all_healthy:
        # Eat a strawberry, mark these teeth as having cavities
        for j in range(i, i + K):
            S[j] = 'X'
        count += 1
        i += K  # Move to the next position after the consumed teeth
    else:
        i += 1  # Move to the next position

print(count)
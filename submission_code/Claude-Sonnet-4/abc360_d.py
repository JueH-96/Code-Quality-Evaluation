N, T = map(int, input().split())
S = input().strip()
X = list(map(int, input().split()))

count = 0

for i in range(N):
    for j in range(i + 1, N):
        # Direction: 1 if moving right (S[i] == '1'), -1 if moving left (S[i] == '0')
        d_i = 1 if S[i] == '1' else -1
        d_j = 1 if S[j] == '1' else -1
        
        # They can only pass each other if moving in opposite directions
        if d_i == d_j:
            continue
            
        # Calculate meeting time
        # At time t: X[i] + d_i * t = X[j] + d_j * t
        # t = (X[j] - X[i]) / (d_i - d_j)
        
        numerator = X[j] - X[i]
        denominator = d_i - d_j
        
        # Check if they meet in the future (t > 0) and within time limit (t ≤ T + 0.1)
        if denominator > 0:
            # t > 0 means numerator > 0
            # t ≤ T + 0.1 means numerator ≤ (T + 0.1) * denominator
            # To avoid floating point, check: 10 * numerator ≤ (10 * T + 1) * denominator
            if numerator > 0 and 10 * numerator <= (10 * T + 1) * denominator:
                count += 1
        elif denominator < 0:
            # t > 0 means numerator < 0
            # t ≤ T + 0.1 means numerator ≥ (T + 0.1) * denominator
            # To avoid floating point, check: 10 * numerator ≥ (10 * T + 1) * denominator
            if numerator < 0 and 10 * numerator >= (10 * T + 1) * denominator:
                count += 1

print(count)
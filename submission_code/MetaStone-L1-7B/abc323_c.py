n, m = map(int, input().split())
A = list(map(int, input().split()))
S = [input().strip() for _ in range(n)]

current_scores = []
for i in range(n):
    total = 0
    for j in range(m):
        if S[i][j] == 'o':
            total += A[j]
    total += (i + 1)
    current_scores.append(total)

for i in range(n):
    # Compute max_other
    max_other = max(current_scores[j] for j in range(n) if j != i)
    
    if current_scores[i] > max_other:
        print(0)
        continue
    
    # Collect unsolved problems for player i
    unsolved = [A[j] for j in range(m) if S[i][j] == 'x']
    unsolved.sort(reverse=True)
    
    # Compute prefix sums
    prefix = [0]
    for num in unsolved:
        prefix.append(prefix[-1] + num)
    
    required = max_other + 1
    for k in range(1, len(unsolved) + 1):
        if current_scores[i] + prefix[k] >= required:
            print(k)
            break
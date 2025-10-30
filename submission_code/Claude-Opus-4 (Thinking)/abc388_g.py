n = int(input())
A = list(map(int, input().split()))
q = int(input())

for _ in range(q):
    L, R = map(int, input().split())
    # Convert to 0-indexed
    L -= 1
    mochi = A[L:R]
    n_mochi = len(mochi)
    used = [False] * n_mochi
    count = 0
    
    for i in range(n_mochi):
        if used[i]:
            continue
        # Try to find the smallest unused base for mochi[i]
        for j in range(i+1, n_mochi):
            if not used[j] and mochi[i] * 2 <= mochi[j]:
                used[i] = True
                used[j] = True
                count += 1
                break
    
    print(count)
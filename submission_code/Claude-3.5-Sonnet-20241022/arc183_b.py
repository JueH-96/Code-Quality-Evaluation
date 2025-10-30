def can_transform(N, K, A, B):
    # For each position i, find what values can be reached from A[i]
    reachable = [set() for _ in range(N)]
    
    # Initial values that can be reached at each position
    for i in range(N):
        reachable[i].add(A[i])
    
    # Keep expanding until no new values can be added
    changed = True
    while changed:
        changed = False
        for i in range(N):
            old_size = len(reachable[i])
            # Try to copy values from positions within K distance
            for j in range(max(0, i-K), min(N, i+K+1)):
                for val in reachable[j]:
                    reachable[i].add(val)
            if len(reachable[i]) > old_size:
                changed = True
    
    # Check if target values are reachable
    for i in range(N):
        if B[i] not in reachable[i]:
            return False
    return True

T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    if can_transform(N, K, A, B):
        print("Yes")
    else:
        print("No")
# YOUR CODE HERE
def can_transform(N, K, A, B):
    groups = {}
    for i in range(N):
        if B[i] not in groups:
            groups[B[i]] = []
        groups[B[i]].append(i)
    
    for i in range(N):
        if A[i] != B[i]:
            if A[i] not in groups:
                return False
            found = False
            for j in groups[A[i]]:
                if abs(i - j) <= K:
                    found = True
                    groups[A[i]].remove(j)
                    if not groups[A[i]]:
                        del groups[A[i]]
                    break
            if not found:
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
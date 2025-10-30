from collections import defaultdict

def solve_test_case():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Create a dictionary to store the indices of each value in A
    indices = defaultdict(list)
    for i, a in enumerate(A):
        indices[a].append(i)

    # Check if it's possible to make A identical to B
    for b in B:
        if b not in indices or len(indices[b]) == 0:
            return "No"
        i = indices[b].pop(0)
        for j in range(max(0, i-K), min(N, i+K+1)):
            if j != i and A[j] == b:
                A[i] = A[j]
                break
        else:
            return "No"

    return "Yes"

T = int(input())
for _ in range(T):
    print(solve_test_case())
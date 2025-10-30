def apply_permutation(A, P):
    N = len(A)
    new_A = [0] * N
    for i in range(N):
        new_A[i] = A[P[i]-1]
    return new_A

def solve():
    N = int(input())
    P = list(map(int, input().split()))
    A = list(map(int, input().split()))
    
    # Find cycle length and all possible permutations
    seen = {}
    curr = A[:]
    all_perms = [curr[:]]
    step = 0
    
    while True:
        curr_tuple = tuple(curr)
        if curr_tuple in seen:
            break
        seen[curr_tuple] = step
        curr = apply_permutation(curr, P)
        all_perms.append(curr[:])
        step += 1
    
    # Find lexicographically smallest
    min_perm = min(all_perms)
    print(*min_perm)

solve()
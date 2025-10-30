# YOUR CODE HERE
import itertools

def solve():
    N, M, K = map(int, input().split())
    keys = [i for i in range(1, N+1)]
    combinations = list(itertools.product([0, 1], repeat=N))
    for _ in range(M):
        C, *A, R = input().split()
        C = int(C)
        A = list(map(int, A))
        for comb in combinations[:]:
            real_keys = [keys[i] for i in range(N) if comb[i] == 1]
            if (R == 'o' and len(set(real_keys) & set(A)) < K) or (R == 'x' and len(set(real_keys) & set(A)) >= K):
                combinations.remove(comb)
    print(len(combinations))

solve()
from collections import defaultdict

def check_equal(A, B, N, K):
    group1, group2 = defaultdict(list), defaultdict(list)
    for i in range(N):
        group1[A[i]].append(i)
        group2[B[i]].append(i)
    for key, positions in group1.items():
        if len(positions) != len(group2.get(key, [])):
            return 'No'
        for pos in positions:
            if all(abs(pos - target) > K for target in group2[key]):
                return 'No'
    return 'Yes'

def solve_equal_sequences():
    T = int(input())
    results = []
    for _ in range(T):
        N, K = map(int, input().split())
        A = list(map(int, input().split()))
        B = list(map(int, input().split()))
        results.append(check_equal(A, B, N, K))
    print('
'.join(results))

solve_equal_sequences()
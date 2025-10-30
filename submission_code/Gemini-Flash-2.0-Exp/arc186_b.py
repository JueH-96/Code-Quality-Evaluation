def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    def check_permutation(p):
        for i in range(n):
            # Condition 1: P_j > P_i for any integer j with A_i < j < i
            for j in range(a[i], i):
                if a[i] < j < i and p[j] < p[i]:
                    return False
            
            # Condition 2: P_{A_i} < P_i if A_i > 0
            if a[i] > 0 and p[a[i]] > p[i]:
                return False
        return True

    import itertools
    
    count = 0
    for p in itertools.permutations(range(1, n + 1)):
        if check_permutation(list(p)):
            count += 1
    
    print(count % 998244353)

solve()
# YOUR CODE HERE
MOD = 998244353

def factorial(n):
    fact = 1
    for i in range(1, n + 1):
        fact = (fact * i) % MOD
    return fact

def solve():
    n, m = map(int, input().split())
    conditions = []
    for _ in range(m):
        conditions.append(list(map(int, input().split())))

    total_permutations = factorial(n)
    
    invalid_permutations = 0
    
    
    def count_invalid_permutations(permutation):
        for l, r, x in conditions:
            sub_permutation = permutation[l-1:r]
            if max(sub_permutation) == permutation[x-1]:
                return True
        return False

    
    import itertools
    
    count = 0
    for permutation_tuple in itertools.permutations(range(1, n + 1)):
        permutation = list(permutation_tuple)
        if count_invalid_permutations(permutation):
            invalid_permutations +=1
    
    print((total_permutations - invalid_permutations + MOD) % MOD)

solve()
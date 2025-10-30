def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))

    def count_inversions(arr):
        count = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    count += 1
        return count

    total_inversions = 0
    num_iterations = n - k + 1
    
    MOD = 998244353

    for i in range(1, n - k + 2):
        sub_array = p[i-1:i+k-1]
        
        import itertools
        
        num_permutations = 0
        inversions_sum = 0
        
        for permutation in itertools.permutations(sub_array):
            num_permutations += 1
            temp_p = p[:]
            temp_p[i-1:i+k-1] = list(permutation)
            inversions_sum += count_inversions(temp_p)
            
        total_inversions = (total_inversions + inversions_sum) % MOD

    
    def mod_inv(a, m):
        return pow(a, m - 2, m)

    expected_value = (total_inversions * mod_inv(num_iterations, MOD) * mod_inv(num_permutations // len(list(itertools.permutations(sub_array))), MOD)) % MOD
    
    total_inversions_all = 0
    for i in range(1, n - k + 2):
        sub_array = p[i-1:i+k-1]
        
        import itertools
        
        num_permutations = 0
        inversions_sum = 0
        
        for permutation in itertools.permutations(sub_array):
            num_permutations += 1
            temp_p = p[:]
            temp_p[i-1:i+k-1] = list(permutation)
            inversions_sum += count_inversions(temp_p)
        total_inversions_all = (total_inversions_all + inversions_sum) % MOD

    expected_value = (total_inversions_all * mod_inv(num_iterations, MOD)) % MOD
    print(expected_value)

solve()
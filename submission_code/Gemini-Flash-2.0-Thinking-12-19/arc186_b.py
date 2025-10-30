def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    def is_valid_permutation(p):
        for i in range(n):
            for j in range(a[i] + 1, i + 1):
                if p[j-1] <= p[i]:
                    return False
            if a[i] > 0:
                if p[a[i]-1] >= p[i]:
                    return False
        return True
        
    def get_valid_permutations_count():
        count = 0
        elements = list(range(1, n + 1))
        
        def get_permutations(current_permutation):
            nonlocal count
            if len(current_permutation) == n:
                if is_valid_permutation(current_permutation):
                    return 1
                else:
                    return 0
            
            current_count = 0
            used_values = set(current_permutation)
            for val in elements:
                if val not in used_values:
                    next_permutation = current_permutation + [val]
                    valid_so_far = True
                    index = len(next_permutation)
                    i = index - 1
                    for j in range(a[i] + 1, i + 1):
                        if next_permutation[j-1] <= next_permutation[i]:
                            valid_so_far = False
                            break
                    if not valid_so_far:
                        continue
                    if a[i] > 0:
                        if next_permutation[a[i]-1] >= next_permutation[i]:
                            valid_so_far = False
                            
                    if valid_so_far:
                        current_count = (current_count + get_permutations(next_permutation)) % 998244353
                        
            return current_count
            
        return get_permutations([])

    result = get_valid_permutations_count()
    print(result)

if __name__ == '__main__':
    solve()
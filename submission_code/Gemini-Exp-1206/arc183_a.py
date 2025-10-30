import math

def solve():
    n, k = map(int, input().split())
    
    total_len = n * k
    
    def count_permutations(counts):
        total_count = 0
        remaining = sum(counts)
        
        if remaining == 0:
            return 1
        
        for i in range(n):
            if counts[i] > 0:
                counts[i] -= 1
                total_count += count_permutations(counts)
                counts[i] += 1
        return total_count

    def find_kth_permutation(kth):
        counts = [k] * n
        result = []
        
        for _ in range(total_len):
            for i in range(n):
                if counts[i] > 0:
                    counts[i] -= 1
                    num_permutations = count_permutations(counts)
                    if kth <= num_permutations:
                        result.append(i + 1)
                        break
                    else:
                        kth -= num_permutations
                        counts[i] += 1
        return result

    total_permutations = count_permutations([k] * n)
    kth_index = (total_permutations + 1) // 2
    
    result = find_kth_permutation(kth_index)
    print(*result)

solve()
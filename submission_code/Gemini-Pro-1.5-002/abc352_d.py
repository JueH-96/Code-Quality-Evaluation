# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))

    min_diff = float('inf')

    for i in range(n - k + 1):
        for j in range(1 << n):
            indices = []
            for bit in range(n):
                if (j >> bit) & 1:
                    indices.append(bit)
            
            if len(indices) == k:
                
                indices.sort()
                
                subsequence = [p[idx] for idx in indices]
                
                
                a_exists = False
                for a in range(1, n - k + 2):
                    
                    expected_set = set(range(a, a + k))
                    subsequence_set = set(subsequence)
                    
                    if expected_set == subsequence_set:
                        a_exists = True
                        break
                
                if a_exists:
                    min_diff = min(min_diff, indices[-1] - indices[0])

    print(min_diff)

solve()
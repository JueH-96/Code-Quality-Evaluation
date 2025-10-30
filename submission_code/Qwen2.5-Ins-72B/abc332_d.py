# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    A = [list(map(int, data[2 + i * W: 2 + (i + 1) * W])) for i in range(H)]
    B = [list(map(int, data[2 + H * W + i * W: 2 + H * W + (i + 1) * W])) for i in range(H)]
    
    def is_possible(A, B):
        from collections import Counter
        return Counter(map(tuple, A)) == Counter(map(tuple, B))
    
    if not is_possible(A, B):
        print(-1)
        return
    
    def min_swaps(A, B):
        from itertools import permutations
        min_ops = float('inf')
        
        for rows_perm in permutations(range(H)):
            for cols_perm in permutations(range(W)):
                transformed_A = [[A[rows_perm[i]][cols_perm[j]] for j in range(W)] for i in range(H)]
                if transformed_A == B:
                    ops = sum(abs(rows_perm[i] - i) for i in range(H)) + sum(abs(cols_perm[j] - j) for j in range(W))
                    min_ops = min(min_ops, ops)
        
        return min_ops
    
    print(min_swaps(A, B))

if __name__ == "__main__":
    main()
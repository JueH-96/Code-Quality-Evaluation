# YOUR CODE HERE
import sys
import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Sort indices in decreasing order of A
    sorted_indices = sorted(range(N), key=lambda x: -A[x])
    
    parent = [i for i in range(N)]
    sum_group = [A[i] for i in range(N)]
    processed = [False]*N
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    for idx in sorted_indices:
        processed[idx] = True
        # Check left neighbor
        if idx > 0 and processed[idx-1]:
            px = find(idx)
            py = find(idx-1)
            if px != py:
                parent[px] = py
                sum_group[py] += sum_group[px]
        # Check right neighbor
        if idx < N-1 and processed[idx+1]:
            px = find(idx)
            py = find(idx+1)
            if px != py:
                parent[px] = py
                sum_group[py] += sum_group[px]
    
    # After processing, for each K, find the sum of its group
    result = []
    for k in range(N):
        root = find(k)
        result.append(str(sum_group[root]))
    
    print(' '.join(result))

if __name__ == "__main__":
    main()
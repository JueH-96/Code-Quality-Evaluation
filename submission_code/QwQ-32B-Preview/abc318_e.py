def count_triples(N, A):
    from collections import defaultdict
    
    # Dictionary to hold lists of positions for each A_i
    pos = defaultdict(list)
    
    for idx, num in enumerate(A, start=1):
        pos[num].append(idx)
    
    total = 0
    
    for indices in pos.values():
        m = len(indices)
        if m >= 2:
            # Compute b[u] = pos[u] - u
            b = [indices[u] - (u + 1) for u in range(m)]
            S = b[0]
            total_sum = 0
            for v in range(1, m):
                total_sum += v * b[v] - S
                S += b[v]
            total += total_sum
    return total

# Read input
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    print(count_triples(N, A))

if __name__ == "__main__":
    main()
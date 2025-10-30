def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = data[1:]
    
    degrees = [0] * (N + 1)
    
    for i in range(0, len(edges), 2):
        u = int(edges[i])
        v = int(edges[i+1])
        degrees[u] += 1
        degrees[v] += 1
    
    centers = [degrees[i] for i in range(1, N+1) if degrees[i] > 1]
    centers.sort()
    
    print(' '.join(map(str, centers)))

if __name__ == '__main__':
    main()
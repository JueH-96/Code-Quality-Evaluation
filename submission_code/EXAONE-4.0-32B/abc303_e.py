def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    deg = [0] * (n + 1)
    index = 1
    for _ in range(n - 1):
        u = int(data[index])
        v = int(data[index + 1])
        index += 2
        deg[u] += 1
        deg[v] += 1
    
    leaves = sum(1 for i in range(1, n + 1) if deg[i] == 1)
    M = (n + 2 - leaves) // 3
    
    centers = []
    for i in range(1, n + 1):
        if deg[i] > 1 and deg[i] != 2:
            centers.append(deg[i])
    
    needed = M - len(centers)
    centers.extend([2] * needed)
    centers.sort()
    print(" ".join(map(str, centers)))

if __name__ == "__main__":
    main()
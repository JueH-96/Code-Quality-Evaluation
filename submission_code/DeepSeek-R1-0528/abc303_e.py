import sys

def main():
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
    M = (n - leaves + 2) // 3
    high_deg_nodes = [deg[i] for i in range(1, n + 1) if deg[i] > 2]
    k = len(high_deg_nodes)
    result = high_deg_nodes + [2] * (M - k)
    result.sort()
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()
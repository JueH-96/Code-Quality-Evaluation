def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    adj = [[] for _ in range(n + 1)]
    index = 1
    for _ in range(n - 1):
        u = int(data[index])
        v = int(data[index + 1])
        adj[u].append(v)
        adj[v].append(u)
        index += 2
    
    max_sum = 0
    for c in range(1, n + 1):
        list_k = []
        for u in adj[c]:
            k = len(adj[u]) - 1
            if k >= 1:
                list_k.append(k)
        list_k.sort(reverse=True)
        for x in range(1, len(list_k) + 1):
            y = list_k[x - 1]
            current_sum = 1 + x + x * y
            if current_sum > max_sum:
                max_sum = current_sum
    print(n - max_sum)

if __name__ == "__main__":
    main()
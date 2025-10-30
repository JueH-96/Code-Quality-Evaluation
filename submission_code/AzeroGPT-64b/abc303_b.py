from collections import defaultdict

def main():
    N, M = map(int, input().split())
    adj = defaultdict(set)
    for _ in range(M):
        prev = 0
        for p in map(int, input().split()):
            if prev:
                adj[prev].add(p)
                adj[p].add(prev)
            prev = p
    ans = N * (N - 1) // 2
    for a in adj.values():
        seen = defaultdict(int)
        for u in a:
            for v in adj[u]:
                if v > u:
                    seen[u, v] += 1 
        for cnt in seen.values():
            ans -= cnt
    print(ans)

if __name__ == "__main__":
    main()
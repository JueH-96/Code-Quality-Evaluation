import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    M = int(input[idx])
    idx +=1

    W = list(map(int, input[idx:idx+N]))
    idx +=N
    A = list(map(int, input[idx:idx+N]))
    idx +=N

    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(input[idx])-1
        idx +=1
        v = int(input[idx])-1
        idx +=1
        adj[u].append(v)
        adj[v].append(u)

    nodes = sorted(range(N), key=lambda x: -W[x])

    f = [0]*N

    for x in nodes:
        total = A[x]
        for y in adj[x]:
            if W[y] < W[x]:
                total += f[y]
        f[x] = total

    print(sum(f))

if __name__ == '__main__':
    main()
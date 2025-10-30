import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    
    # Initialize adjacency list
    adj = defaultdict(set)
    
    for _ in range(Q):
        type_q = int(data[idx])
        if type_q == 1:
            u = int(data[idx+1])
            v = int(data[idx+2])
            adj[u].add(v)
            adj[v].add(u)
            idx += 3
        else:
            v = int(data[idx+1])
            k = int(data[idx+2])
            connected = sorted(adj[v], reverse=True)
            if len(connected) >= k:
                print(connected[k-1])
            else:
                print(-1)
            idx += 3

if __name__ == "__main__":
    main()
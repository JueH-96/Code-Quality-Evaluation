# YOUR CODE HERE
import sys
def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    Q = int(input[idx]); idx +=1
    degrees = [0]*(N+1)
    adj = [set() for _ in range(N+1)]
    zero_count = N
    res = []
    for _ in range(Q):
        query_type = input[idx]; idx +=1
        if query_type == '1':
            u = int(input[idx]); idx +=1
            v = int(input[idx]); idx +=1
            if degrees[u] ==0:
                zero_count -=1
            if degrees[v] ==0:
                zero_count -=1
            degrees[u] +=1
            degrees[v] +=1
            adj[u].add(v)
            adj[v].add(u)
            res.append(str(zero_count))
        else:
            v = int(input[idx]); idx +=1
            if degrees[v] >0:
                zero_count +=1
                neighbors = list(adj[v])
                for neighbor in neighbors:
                    adj[neighbor].remove(v)
                    degrees[neighbor] -=1
                    if degrees[neighbor] ==0:
                        zero_count +=1
                degrees[v] =0
                adj[v].clear()
            res.append(str(zero_count))
    print('
'.join(res))
if __name__ == '__main__':
    main()
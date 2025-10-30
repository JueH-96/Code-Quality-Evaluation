import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    X = int(data[idx+1])
    idx +=2
    
    A = list(map(int, data[idx:idx+N]))
    idx +=N
    
    B = list(map(int, data[idx:idx+N]))
    idx +=N
    
    P = list(map(int, data[idx:idx+N]))
    idx +=N
    
    Q = list(map(int, data[idx:idx+N]))
    idx +=N
    
    def get_cycle(perm, start):
        cycle = set()
        current = start
        while True:
            if current in cycle:
                break
            cycle.add(current)
            current = perm[current-1]  # perm is 0-based list, nodes are 1-based
        return cycle
    
    S_p = get_cycle(P, X)
    S_q = get_cycle(Q, X)
    
    possible = True
    for i in range(N):
        node = i + 1
        if A[i] > 0 and node not in S_p:
            possible = False
        if B[i] > 0 and node not in S_q:
            possible = False
    
    if not possible:
        print(-1)
        return
    
    total_balls = sum(A) + sum(B)
    if total_balls == 0:
        print(0)
        return
    
    union = S_p.union(S_q)
    print(len(union) - 1)

if __name__ == '__main__':
    main()
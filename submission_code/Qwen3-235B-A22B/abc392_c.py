import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    Q = list(map(int, data[N+1:2*N+1]))
    
    Q_map = {q: i+1 for i, q in enumerate(Q)}
    
    S = []
    for i in range(1, N+1):
        k = Q_map[i]
        p = P[k-1]
        s_i = Q[p-1]
        S.append(str(s_i))
    
    print(' '.join(S))

if __name__ == '__main__':
    main()
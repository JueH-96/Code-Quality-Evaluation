def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    P = list(map(int, input[idx:idx+N]))
    idx += N
    Q = list(map(int, input[idx:idx+N]))
    idx += N
    
    pos_p = [0] * (N + 1)
    for i in range(N):
        pos_p[P[i]] = i + 1
    
    pos_q = [0] * (N + 1)
    for i in range(N):
        pos_q[Q[i]] = i + 1
    
    for x in range(1, N + 1):
        row = []
        for y in range(1, N + 1):
            if pos_p[x] <= (N - pos_q[y]):
                row.append('0')
            else:
                row.append('1')
        print(''.join(row))
    
if __name__ == '__main__':
    main()
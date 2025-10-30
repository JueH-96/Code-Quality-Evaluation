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

    pos_P = {x: i for i, x in enumerate(P)}
    pos_Q = {x: i for i, x in enumerate(Q)}

    for i in range(N):
        row = []
        for j in range(N):
            p_i = pos_P[i+1]
            q_j = pos_Q[j+1]
            p_j = pos_P[j+1]
            q_i = pos_Q[i+1]
            if p_i < p_j and q_j < q_i:
                row.append('0')
            else:
                row.append('1')
        print(''.join(row))

if __name__ == '__main__':
    main()
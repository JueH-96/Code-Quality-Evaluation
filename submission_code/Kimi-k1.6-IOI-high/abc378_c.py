def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    last_pos = dict()
    B = []
    for i in range(N):
        val = A[i]
        pos = i + 1
        if val in last_pos:
            B.append(last_pos[val])
        else:
            B.append(-1)
        last_pos[val] = pos
    print(' '.join(map(str, B)))

if __name__ == "__main__":
    main()
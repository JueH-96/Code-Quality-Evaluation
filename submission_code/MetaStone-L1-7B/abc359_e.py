def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    steps = [0] * N
    current = 0
    for i in range(N):
        hi = H[i]
        if i == 0:
            m = max(hi, current)
        else:
            m = max(hi, current)
        current = m + 1
        steps[i] = current
    print(' '.join(map(str, steps)))

if __name__ == '__main__':
    main()
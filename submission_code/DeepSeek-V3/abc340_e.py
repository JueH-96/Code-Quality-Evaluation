def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    for b in B:
        balls = A[b]
        A[b] = 0
        C = 0
        while balls > 0:
            idx = (b + C) % N
            A[idx] += 1
            balls -= 1
            C += 1
    
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()
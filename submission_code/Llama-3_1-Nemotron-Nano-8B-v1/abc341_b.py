def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    for i in range(N-1):
        S = int(input[ptr])
        T = int(input[ptr+1])
        ptr +=2
        S_i = S
        T_i = T
        # Process country i (0-based)
        if S_i ==0:
            continue  # but S_i >=1 per constraints
        k = A[i] // S_i
        converted = k * T_i
        A[i] -= k * S_i
        A[i+1] += converted
    print(A[-1])

if __name__ == '__main__':
    main()
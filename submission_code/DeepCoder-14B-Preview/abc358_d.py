def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+M]))
    
    A.sort()
    B.sort()
    
    possible = True
    for i in range(M):
        if A[i] < B[i]:
            possible = False
            break
    
    if not possible:
        print(-1)
    else:
        print(sum(A[:M]))

if __name__ == '__main__':
    main()
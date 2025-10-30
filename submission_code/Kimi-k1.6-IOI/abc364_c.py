def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    X = int(input[ptr])
    ptr +=1
    Y = int(input[ptr])
    ptr +=1
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    B = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    
    # Compute k1
    sorted_A = sorted(A, reverse=True)
    prefix_A = [0]
    for a in sorted_A:
        prefix_A.append(prefix_A[-1] + a)
    k1 = None
    for i in range(1, N+1):
        if prefix_A[i] > X:
            k1 = i
            break
    else:
        k1 = N +1
    
    # Compute k2
    sorted_B = sorted(B, reverse=True)
    prefix_B = [0]
    for b in sorted_B:
        prefix_B.append(prefix_B[-1] + b)
    k2 = None
    for i in range(1, N+1):
        if prefix_B[i] > Y:
            k2 = i
            break
    else:
        k2 = N +1
    
    ans = min(k1, k2)
    if ans <= N:
        print(ans)
    else:
        print(N)

if __name__ == '__main__':
    main()
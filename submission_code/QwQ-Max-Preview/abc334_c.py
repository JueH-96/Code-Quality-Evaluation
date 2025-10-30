def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:2+K]))
    
    if K == 0:
        print(0)
        return
    
    A.sort()
    if K == 1:
        print(0)
        return
    
    if K % 2 == 0:
        total = 0
        for i in range(0, K, 2):
            total += A[i+1] - A[i]
    else:
        total = A[-1] - A[0]
        max_diff = 0
        for i in range(K-1):
            current_diff = A[i+1] - A[i]
            if current_diff > max_diff:
                max_diff = current_diff
        total -= max_diff
    print(total)

if __name__ == '__main__':
    main()
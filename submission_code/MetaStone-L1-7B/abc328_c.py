def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    S = data[idx]
    idx += 1

    # Compute array A
    A = []
    for i in range(len(S) - 1):
        if S[i] == S[i+1]:
            A.append(1)
        else:
            A.append(0)
    
    # Compute prefix sum array B
    B = [0] * (len(A) + 1)
    for i in range(len(A)):
        B[i+1] = B[i] + A[i]

    # Process each query
    for _ in range(Q):
        l = int(data[idx])
        idx += 1
        r = int(data[idx])
        idx += 1
        if l >= r:
            print(0)
            continue
        a = l - 1
        b = r - 2
        if a >= len(A) or b >= len(A):
            print(0)
            continue
        if a < 0 or b < 0:
            print(0)
            continue
        if a > b:
            print(0)
            continue
        ans = B[b+1] - B[a]
        print(ans)

if __name__ == '__main__':
    main()
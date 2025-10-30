def next_sequence(A, R):
    n = len(A)
    for i in range(n-1, -1, -1):
        if A[i] < R[i]:
            A[i] += 1
            return True
        else:
            A[i] = 1
            if i == 0:
                return False
    return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    R = list(map(int, data[2:2+N]))
    
    A = [1] * N
    while True:
        sum_A = sum(A)
        if sum_A % K == 0:
            print(' '.join(map(str, A)))
        if not next_sequence(A, R):
            break

if __name__ == "__main__":
    main()
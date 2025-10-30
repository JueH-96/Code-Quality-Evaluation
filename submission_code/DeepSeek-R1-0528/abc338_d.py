import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    X = list(map(int, data[2:2+m]))
    X0 = [x-1 for x in X]

    F = [0] * (n + 1)
    
    for i in range(m - 1):
        a = X0[i]
        b = X0[i + 1]
        A = a - 1
        if A < 0:
            A += n
        B = b - 1
        if B < 0:
            B += n
            
        L = min(A, B)
        R = max(A, B)
        d = abs(A - B)
        term1 = d
        term2 = n - d
        
        F[0] += term1
        if L + 1 < n:
            F[L + 1] += term2 - term1
        if R + 1 < n:
            F[R + 1] += term1 - term2

    current = F[0]
    min_val = current
    for i in range(1, n):
        current += F[i]
        if current < min_val:
            min_val = current
    print(min_val)

if __name__ == "__main__":
    main()
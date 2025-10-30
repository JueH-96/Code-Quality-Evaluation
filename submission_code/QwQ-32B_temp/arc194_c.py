import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))
    
    S = []
    sum_A_nonS = 0
    for i in range(N):
        a = A[i]
        b = B[i]
        if a != b:
            S.append((a, C[i]))
        else:
            sum_A_nonS += a * C[i]
    
    m = len(S)
    if m == 0:
        print(0)
        return
    
    slopes = []
    sum_S_part = 0
    for a, c in S:
        slope = (2 * a - 1) * c
        slopes.append(slope)
        term = c * ((m + 1) * (1 - a) - a)
        sum_S_part += term
    
    slopes.sort()
    sum_slope = 0
    for i in range(m):
        t = m - i
        sum_slope += slopes[i] * t
    
    fixed_total = sum_A_nonS * m + sum_S_part
    total = fixed_total + sum_slope
    print(total)

if __name__ == "__main__":
    main()
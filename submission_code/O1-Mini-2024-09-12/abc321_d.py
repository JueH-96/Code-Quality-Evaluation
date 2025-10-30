import sys
import bisect

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    N, M, P = map(int, data[:3])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))
    
    B.sort()
    prefix_B = [0]*(M+1)
    for i in range(M):
        prefix_B[i+1] = prefix_B[i] + B[i]
    
    total = 0
    for a in A:
        # Find the largest j where a + B[j] <= P
        # which is equivalent to B[j] <= P - a
        limit = P - a
        if limit < 0:
            k = 0
        else:
            k = bisect.bisect_right(B, limit)
        sum_B = prefix_B[k]
        total += a * k + sum_B + (M - k) * P
    print(total)

if __name__ == "__main__":
    main()
import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    P = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    ptr += M
    
    B.sort()
    prefix = [0] * (M + 1)
    for i in range(M):
        prefix[i+1] = prefix[i] + B[i]
    
    total = 0
    for a in A:
        threshold = P - a
        cnt = bisect.bisect_right(B, threshold)
        sum1 = a * cnt + prefix[cnt]
        sum2 = (M - cnt) * P
        total += sum1 + sum2
    
    print(total)

if __name__ == '__main__':
    main()
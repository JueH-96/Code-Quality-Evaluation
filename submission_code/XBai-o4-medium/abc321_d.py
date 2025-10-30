import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    P = int(input[idx])
    idx += 1
    
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+M]))
    
    sorted_B = sorted(B)
    prefix = [0] * (M + 1)
    for i in range(M):
        prefix[i+1] = prefix[i] + sorted_B[i]
    
    total = 0
    for a in A:
        target = P - a
        k = bisect.bisect_right(sorted_B, target)
        contrib = a * k + prefix[k] + (M - k) * P
        total += contrib
    
    print(total)

if __name__ == "__main__":
    main()
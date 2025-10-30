import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    S = sorted(A)
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + S[i]
    total = prefix[N]
    
    result = []
    for a in A:
        j = bisect.bisect_right(S, a)
        sum_val = total - prefix[j]
        result.append(str(sum_val))
    print(' '.join(result))

if __name__ == "__main__":
    main()
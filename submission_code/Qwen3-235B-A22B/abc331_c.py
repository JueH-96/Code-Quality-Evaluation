import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    A = list(map(int, data[1:n+1]))
    S = sorted(A)
    
    # Compute suffix sums
    suffix_sums = [0] * (n + 1)
    for i in range(n-1, -1, -1):
        suffix_sums[i] = suffix_sums[i+1] + S[i]
    
    # Prepare result
    result = []
    for x in A:
        k = bisect.bisect_right(S, x)
        result.append(str(suffix_sums[k]))
    print(' '.join(result))

if __name__ == "__main__":
    main()
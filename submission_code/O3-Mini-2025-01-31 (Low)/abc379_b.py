def main():
    import sys
    input = sys.stdin.readline

    N, K = map(int, input().split())
    S = list(input().strip())
    
    count = 0
    i = 0
    while i <= N - K:
        # Check if K consecutive healthy teeth starting at index i
        if all(S[j] == 'O' for j in range(i, i + K)):
            # We can eat a strawberry using these K teeth.
            count += 1
            # Turn these teeth unhealthy.
            for j in range(i, i + K):
                S[j] = 'X'
            # Move pointer i forward by K because these teeth have become used.
            i += K
        else:
            i += 1
    
    print(count)

if __name__ == '__main__':
    main()
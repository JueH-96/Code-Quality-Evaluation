def main():
    import sys
    input = sys.stdin.readline
    
    N, K = map(int, input().split())
    S = list(input().strip())
    
    count = 0
    # Scan from left to right; whenever we find K consecutive 'O's, "use" them
    for i in range(N - K + 1):
        # Check if the segment [i, i+K) is all 'O'
        if all(S[j] == 'O' for j in range(i, i + K)):
            count += 1
            # Mark them as used (cavities/vowels) by setting to 'X'
            for j in range(i, i + K):
                S[j] = 'X'
    print(count)

if __name__ == "__main__":
    main()
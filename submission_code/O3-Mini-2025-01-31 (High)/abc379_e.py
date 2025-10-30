def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].strip()
    
    total = 0
    dp = 0
    # Iterate over positions 1 to N (1-indexed)
    for i in range(1, N + 1):
        # Convert the i-th character (S is 0-indexed) to an integer.
        d = int(S[i - 1])
        # The recurrence relation:
        # dp[i] = dp[i-1]*10 + i * (digit at position i)
        dp = dp * 10 + i * d
        total += dp
    sys.stdout.write(str(total))

if __name__ == '__main__':
    main()
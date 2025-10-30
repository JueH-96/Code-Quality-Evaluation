def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    S = list(data[2])

    ans = 0
    i = 0
    while i <= N - K:
        # Check if the next K teeth are all healthy (i.e., 'O')
        if all(ch == 'O' for ch in S[i:i+K]):
            ans += 1
            # Mark those K consecutive teeth as unhealthy ('X') after eating
            for j in range(i, i+K):
                S[j] = 'X'
            # Jump past these K teeth
            i += K
        else:
            i += 1

    print(ans)

# Do not forget to call main()!
main()
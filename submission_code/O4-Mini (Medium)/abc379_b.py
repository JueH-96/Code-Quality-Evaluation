def main():
    import sys
    data = sys.stdin.read().split()
    N, K = map(int, data[:2])
    S = data[2].strip()
    
    ans = 0
    i = 0
    # scan from left to right
    while i + K <= N:
        # if there's a run of K healthy teeth starting at i
        if S[i:i+K] == 'O'*K:
            ans += 1
            # consume these K teeth (they become cavities), so skip them
            i += K
        else:
            i += 1
    
    print(ans)

if __name__ == "__main__":
    main()
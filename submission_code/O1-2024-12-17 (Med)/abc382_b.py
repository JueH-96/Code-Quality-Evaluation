def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, D = map(int, data[:2])
    S = list(data[2])
    
    # Remove D cookies from the rightmost to leftmost
    removed = 0
    for i in range(N - 1, -1, -1):
        if S[i] == '@' and removed < D:
            S[i] = '.'
            removed += 1
        if removed == D:
            break
    
    print("".join(S))

if __name__ == "__main__":
    main()
def main():
    import sys
    data = sys.stdin.read().split()
    S = data[0]
    T = data[1]
    n = len(S)
    ans = []
    j = 0
    for i, ch in enumerate(T):
        if j < n and ch == S[j]:
            ans.append(str(i+1))
            j += 1
            if j == n:
                break
    # Print positions of correctly typed characters
    print(" ".join(ans))

if __name__ == "__main__":
    main()
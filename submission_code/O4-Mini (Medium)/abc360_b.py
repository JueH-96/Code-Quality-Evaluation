def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        print("No")
        return
    S, T = data[0], data[1]
    N = len(S)
    M = len(T)
    # Try all w in [1, N-1] and c in [1, w]
    for w in range(1, N):
        for c in range(1, w+1):
            res_chars = []
            # Split S into chunks of size w
            for i in range(0, N, w):
                # chunk is S[i:i+w]
                # if chunk length >= c, take c-th character
                if i + c - 1 < min(i + w, N):
                    res_chars.append(S[i + c - 1])
            # Join and compare
            if "".join(res_chars) == T:
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()
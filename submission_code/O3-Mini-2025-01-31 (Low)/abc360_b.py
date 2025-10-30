def main():
    import sys
    input = sys.stdin.readline
    S, T = sys.stdin.read().split()
    n = len(S)
    # Since w must be in [1, n-1]
    for w in range(1, n):
        # For c in [1, w]
        for c in range(1, w+1):
            result = []
            index = 0
            while index < n:
                chunk = S[index:index + w]
                if len(chunk) >= c:  # only append if chunk length is at least c
                    result.append(chunk[c-1])
                index += w
            if "".join(result) == T:
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()
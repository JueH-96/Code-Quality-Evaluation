def main():
    import sys
    data = sys.stdin.read().split()
    S, T = data[0], data[1]
    lenS = len(S)
    lenT = len(T)

    # We want to find 1 <= c <= w < |S|
    for w in range(1, lenS):          # w from 1 to lenS-1
        for c in range(1, w + 1):     # c from 1 to w
            chars = []
            # Split S in chunks of length w
            for i in range(0, lenS, w):
                chunk = S[i:i+w]
                if len(chunk) >= c:
                    # c-th character in chunk (1-based => c-1 index)
                    chars.append(chunk[c-1])
            if "".join(chars) == T:
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()
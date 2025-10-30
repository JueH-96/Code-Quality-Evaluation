def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        print("No")
        return
    S = data[0]
    T = data[1]
    n = len(S)

    # Try all possible w and c
    for w in range(1, n):          # w from 1 to n-1
        for c in range(1, w+1):    # c from 1 to w
            res_chars = []
            # split S into chunks of size w
            for i in range(0, n, w):
                chunk = S[i:i+w]
                # only if chunk length >= c we take its c-th char
                if len(chunk) >= c:
                    res_chars.append(chunk[c-1])
            # build result
            if "".join(res_chars) == T:
                print("Yes")
                return

    print("No")


if __name__ == "__main__":
    main()
def solve():
    import sys
    data = sys.stdin.read().strip().split()
    S, T = data[0], data[1]
    lenS = len(S)
    lenT = len(T)

    # We need to find c, w such that 1 <= c <= w < |S|,
    # and if we partition S into chunks of length w,
    # then concatenating the c-th characters of those chunks
    # (provided they have length >= c) equals T.

    # Iterate over all possible w
    for w in range(1, lenS):
        # Iterate over all possible c
        for c in range(1, w + 1):
            # Build the string by taking the c-th character of each chunk
            chunks = []
            for start in range(0, lenS, w):
                chunk = S[start:start+w]
                if len(chunk) >= c:
                    chunks.append(chunk[c-1])
            # Compare the concatenated characters with T
            if "".join(chunks) == T:
                print("Yes")
                return

    # If nothing matched, print No
    print("No")

def main():
    solve()

if __name__ == "__main__":
    main()
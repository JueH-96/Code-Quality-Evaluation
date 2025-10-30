import sys

def main() -> None:
    S, T = sys.stdin.read().strip().split()
    n, m = len(S), len(T)

    # Try every possible pair (c, w)
    for w in range(1, n):          # w must be < |S|
        for c in range(1, w + 1):  # 1 ≤ c ≤ w
            # How many substrings will contribute a character?
            full_chunks = n // w                 # all of them have length w ≥ c
            cnt = full_chunks
            if n % w >= c:                       # last (possibly shorter) chunk
                cnt += 1

            # Length must match |T|
            if cnt != m:
                continue

            # Build the resulting string
            res = []
            for i in range(0, n, w):
                chunk = S[i:i + w]
                if len(chunk) >= c:
                    res.append(chunk[c - 1])
            if ''.join(res) == T:
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()
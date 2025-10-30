import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n = data[0]
    a = data[1:]

    p = -1           # largest index that can be put on top of current bottom
    ans = 0

    for j in range(n):                    # j : index of the bottom mochi
        # extend p while next candidate still satisfies 2 * a[i] ≤ a[j]
        while p + 1 < j and 2 * a[p + 1] <= a[j]:
            p += 1
        # all indices 0..p (inclusive) are valid tops for bottom j
        ans += p + 1                      # p is -1 when no one fits

    print(ans)

if __name__ == "__main__":
    main()
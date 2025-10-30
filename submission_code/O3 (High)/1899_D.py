import sys
from collections import Counter

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    t = data[0]
    pos = 1                  # current reading position in data
    out_lines = []

    for _ in range(t):
        n = data[pos]
        pos += 1
        arr = data[pos:pos + n]
        pos += n

        cnt = Counter(arr)

        # pairs with equal exponents
        ans = sum(c * (c - 1) // 2 for c in cnt.values())

        # pairs {1, 2}
        ans += cnt.get(1, 0) * cnt.get(2, 0)

        out_lines.append(str(ans))

    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()
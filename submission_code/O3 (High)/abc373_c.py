import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n = data[0]
    a_vals = data[1:1 + n]
    b_vals = data[1 + n:1 + 2 * n]

    max_a = max(a_vals)
    max_b = max(b_vals)

    print(max_a + max_b)

if __name__ == "__main__":
    main()
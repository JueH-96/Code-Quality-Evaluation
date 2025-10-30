import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    total_A = 0
    max_diff = -10**18  # sufficiently small
    idx = 1
    for _ in range(n):
        A = int(data[idx]); B = int(data[idx + 1])
        idx += 2
        total_A += A
        diff = B - A
        if diff > max_diff:
            max_diff = diff
    # Maximum head height
    ans = total_A + max_diff
    print(ans)

if __name__ == "__main__":
    main()
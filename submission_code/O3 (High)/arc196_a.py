import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    a = list(map(int, data[1:]))

    a.sort()
    k = n // 2

    # sum of largest k minus sum of smallest k
    ans = sum(a[-k:]) - sum(a[:k])
    print(ans)

if __name__ == "__main__":
    main()
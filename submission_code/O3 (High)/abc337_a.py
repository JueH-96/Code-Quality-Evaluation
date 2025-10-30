import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    takahashi_total = 0
    aoki_total = 0

    idx = 1
    for _ in range(n):
        x = int(data[idx]); y = int(data[idx + 1])
        idx += 2
        takahashi_total += x
        aoki_total += y

    if takahashi_total > aoki_total:
        print("Takahashi")
    elif takahashi_total < aoki_total:
        print("Aoki")
    else:
        print("Draw")

if __name__ == "__main__":
    main()
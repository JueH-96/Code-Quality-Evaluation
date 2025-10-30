import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n = data[0]
    a = data[1:1 + n]

    total = 0          # S_N
    min_prefix = 0     # min{0, S_1, … , S_N}

    for v in a:
        total += v
        if total < min_prefix:
            min_prefix = total

    # smallest initial passengers x = max(0, -min_prefix)
    answer = max(0, -min_prefix) + total
    print(answer)

if __name__ == "__main__":
    main()
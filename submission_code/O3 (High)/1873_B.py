import sys

def main() -> None:
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    idx = 1
    res = []
    for _ in range(t):
        n = int(data[idx]); idx += 1
        digits = list(map(int, data[idx: idx + n])); idx += n

        best = 0
        for i in range(n):
            prod = digits[i] + 1          # increment chosen digit
            for j in range(n):
                if j != i:
                    prod *= digits[j]
            if prod > best:
                best = prod
        res.append(str(best))

    sys.stdout.write("
".join(res))

if __name__ == "__main__":
    main()
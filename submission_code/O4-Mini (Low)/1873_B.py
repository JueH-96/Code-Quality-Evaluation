import sys
import threading

def main():
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    out_lines = []
    for _ in range(t):
        n = int(data[idx]); idx += 1
        a = list(map(int, data[idx:idx+n]))
        idx += n
        best = 0
        # Try adding 1 to each position
        for i in range(n):
            prod = 1
            for j in range(n):
                if i == j:
                    prod *= (a[j] + 1)
                else:
                    prod *= a[j]
            if prod > best:
                best = prod
        out_lines.append(str(best))
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()
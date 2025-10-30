import sys
import threading

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    idx = 1
    out_lines = []
    for _ in range(t):
        n = int(data[idx]); idx += 1
        cnt = {}
        for i in range(n):
            a = int(data[idx]); idx += 1
            cnt[a] = cnt.get(a, 0) + 1
        # identical pairs
        total = 0
        for v in cnt.values():
            total += v * (v - 1) // 2
        # cross pairs (1,2)
        total += cnt.get(1, 0) * cnt.get(2, 0)
        out_lines.append(str(total))
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()
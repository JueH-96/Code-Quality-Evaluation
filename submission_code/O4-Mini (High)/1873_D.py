import sys
import threading

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    idx = 1
    out = []
    for _ in range(t):
        n = int(data[idx]); k = int(data[idx+1])
        idx += 2
        s = data[idx]
        idx += 1

        covered_end = -1
        ops = 0
        # Greedily cover each black cell that lies beyond the current coverage
        for i, ch in enumerate(s):
            if ch == 'B' and i > covered_end:
                # Place the k-interval as far right as possible so as to cover i
                start = min(i, n - k)
                covered_end = start + k - 1
                ops += 1

        out.append(str(ops))

    sys.stdout.write("
".join(out))


if __name__ == "__main__":
    main()
import sys
import threading

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    idx = 1
    out = []
    for _ in range(t):
        n = int(data[idx]); k = int(data[idx+1]); idx += 2
        s = data[idx]; idx += 1
        ans = 0
        i = 0
        # Greedy: whenever we see a 'B', perform an operation covering [i, i+k-1]
        # and skip past these k cells.
        while i < n:
            if s[i] == 'B':
                ans += 1
                i += k
            else:
                i += 1
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
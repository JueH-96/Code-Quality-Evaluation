import sys
import threading

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    idx = 1
    out = []
    for _ in range(t):
        n = int(data[idx]); k = int(data[idx+1])
        idx += 2
        A = data[idx:idx+n]
        idx += n
        B = data[idx:idx+n]
        idx += n
        # Since K >= 1, the "copy within distance K" graph is connected,
        # so any value present at least once in A can be copied arbitrarily.
        sA = set(A)
        ok = True
        for b in B:
            if b not in sA:
                ok = False
                break
        out.append("Yes" if ok else "No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
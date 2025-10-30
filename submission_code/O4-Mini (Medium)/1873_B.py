import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    t = int(next(it))
    out = []
    for _ in range(t):
        n = int(next(it))
        a = [int(next(it)) for _ in range(n)]
        maxp = 0
        for i in range(n):
            p = 1
            for j, x in enumerate(a):
                if j == i:
                    p *= (x + 1)
                else:
                    p *= x
            if p > maxp:
                maxp = p
        out.append(str(maxp))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
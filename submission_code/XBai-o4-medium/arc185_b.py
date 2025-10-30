import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    t = int(data[ptr])
    ptr += 1
    for _ in range(t):
        n = int(data[ptr])
        ptr += 1
        a = list(map(int, data[ptr:ptr + n]))
        ptr += n
        s = sum(a)
        prefix = 0
        ok = True
        for i in range(1, n):
            prefix += a[i-1]
            if prefix * n > s * i:
                ok = False
                break
        print("Yes" if ok else "No")

if __name__ == "__main__":
    main()
import sys

def main():
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    res = []
    j = 0
    for ch in S:
        while j < len(T) and T[j] != ch:
            j += 1
        res.append(j + 1)
        j += 1
    print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()
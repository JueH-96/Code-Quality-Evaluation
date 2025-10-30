import sys

def main():
    n = int(sys.stdin.readline())
    seen = set()
    for _ in range(n):
        s = sys.stdin.readline().strip()
        rev = s[::-1]
        key = min(s, rev)
        seen.add(key)
    print(len(seen))

if __name__ == "__main__":
    main()
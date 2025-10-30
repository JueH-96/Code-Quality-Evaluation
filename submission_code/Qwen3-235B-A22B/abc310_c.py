import sys

def main():
    n = int(sys.stdin.readline())
    unique = set()
    for _ in range(n):
        s = sys.stdin.readline().strip()
        rev = s[::-1]
        key = min(s, rev)
        unique.add(key)
    print(len(unique))

if __name__ == "__main__":
    main()
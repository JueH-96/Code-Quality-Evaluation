import sys

def main():
    input = sys.stdin.readline
    n = int(input().strip())
    seen = set()
    for _ in range(n):
        s = input().strip()
        # Compute the canonical form: the lexicographically smaller
        # between s and its reversal
        rev = s[::-1]
        canon = s if s <= rev else rev
        seen.add(canon)
    print(len(seen))

if __name__ == "__main__":
    main()
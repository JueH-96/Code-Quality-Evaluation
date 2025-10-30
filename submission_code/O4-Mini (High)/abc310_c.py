import sys

def main():
    input = sys.stdin.readline
    n = int(input())
    seen = set()
    for _ in range(n):
        s = input().rstrip()
        # Use the lexicographically smaller of s and its reverse as the canonical form
        seen.add(s if s <= s[::-1] else s[::-1])
    print(len(seen))

main()
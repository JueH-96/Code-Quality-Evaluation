def main():
    import sys
    input = sys.stdin.readline
    t = int(input())
    target = "abc"
    for _ in range(t):
        s = input().strip()
        # Count how many positions differ from "abc"
        mismatches = sum(1 for i in range(3) if s[i] != target[i])
        # We can fix at most two mismatches with one swap (or zero swaps if already equal)
        print("YES" if mismatches <= 2 else "NO")

if __name__ == "__main__":
    main()
def main():
    import sys

    input = sys.stdin.readline
    n = int(input().strip())
    s = input().strip()

    # Length must be odd
    if n % 2 == 0:
        print("No")
        return

    mid = n // 2

    # Middle character must be '/'
    if s[mid] != '/':
        print("No")
        return

    # Left half must be all '1'
    left = s[:mid]
    if any(ch != '1' for ch in left):
        print("No")
        return

    # Right half must be all '2'
    right = s[mid+1:]
    if any(ch != '2' for ch in right):
        print("No")
        return

    print("Yes")

if __name__ == "__main__":
    main()
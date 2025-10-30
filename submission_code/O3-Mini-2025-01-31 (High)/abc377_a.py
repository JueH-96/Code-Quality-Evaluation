def main():
    import sys
    input = sys.stdin.readline
    S = input().strip()
    if sorted(S) == list("ABC"):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
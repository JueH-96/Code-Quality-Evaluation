def main():
    import sys

    S = sys.stdin.readline().strip()
    # Check if S contains exactly one "A", one "B", and one "C"
    if sorted(S) == sorted("ABC"):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
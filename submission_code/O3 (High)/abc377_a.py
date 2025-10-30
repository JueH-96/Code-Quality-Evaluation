import sys

def main():
    S = sys.stdin.readline().strip()
    print("Yes" if "".join(sorted(S)) == "ABC" else "No")

if __name__ == "__main__":
    main()
import sys

def main():
    S = sys.stdin.readline().strip()
    print("Yes" if S.endswith("san") else "No")

if __name__ == "__main__":
    main()
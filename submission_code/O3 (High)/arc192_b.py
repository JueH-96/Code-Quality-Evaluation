import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    if N & 1:
        print("Fennec")
    else:
        print("Snuke")

if __name__ == "__main__":
    main()
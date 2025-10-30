import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:1+N]))

    for i in range(N - 1):
        if A[i] >= A[i + 1]:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()
import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:1+N]))

    for i in range(N - 2):
        if A[i] == A[i + 1] == A[i + 2]:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()
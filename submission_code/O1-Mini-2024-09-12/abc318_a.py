import sys

def main():
    N, M, P = map(int, sys.stdin.read().split())
    if M > N:
        print(0)
    else:
        count = (N - M) // P + 1
        print(count)

if __name__ == "__main__":
    main()
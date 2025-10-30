import sys

def main():
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    P = int(data[2])

    # If the first full moon day is after N, he sees none.
    if M > N:
        print(0)
        return

    # Count how many terms in the AP M, M+P, M+2P, ... are <= N.
    count = (N - M) // P + 1
    print(count)

if __name__ == "__main__":
    main()
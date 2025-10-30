def main():
    import sys
    data = sys.stdin.readline().split()
    if not data:
        return
    N, M, P = map(int, data)
    if M > N:
        print(0)
    else:
        # Count how many terms of the arithmetic sequence M, M+P, ... are ≤ N
        print((N - M) // P + 1)

main()
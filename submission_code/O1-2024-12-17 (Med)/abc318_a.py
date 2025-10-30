def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M, P = map(int, data)

    if M > N:
        print(0)
    else:
        # The count of full moons is 1 + the number of times we can add P 
        # before exceeding N starting from M
        count = 1 + (N - M) // P
        print(count)

# Do not forget to call main.
main()
def main():
    import sys
    X = int(sys.stdin.readline().strip())
    factorial = 1
    n = 1
    # Loop until factorial matches X
    while factorial < X:
        n += 1
        factorial *= n
    # At this point factorial == X by problem guarantee
    print(n)

if __name__ == "__main__":
    main()
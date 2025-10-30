def main():
    X = int(input().strip())
    f = 1
    n = 1
    # Keep multiplying until we reach X
    while f < X:
        n += 1
        f *= n
    # By problem guarantee, f == X here
    print(n)

if __name__ == "__main__":
    main()
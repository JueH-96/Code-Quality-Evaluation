def main():
    N = int(input().strip())
    rem = N % 5
    # distance to the lower multiple of 5
    d1 = rem
    # distance to the higher multiple of 5
    d2 = 0 if rem == 0 else 5 - rem

    if d1 <= d2:
        print(N - d1)
    else:
        print(N + d2)

if __name__ == "__main__":
    main()
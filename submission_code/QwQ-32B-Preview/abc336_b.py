def ctz(N):
    count = 0
    while (N & 1) == 0 and N != 0:
        count += 1
        N = N >> 1
    return count

def main():
    N = int(input().strip())
    print(ctz(N))

if __name__ == "__main__":
    main()
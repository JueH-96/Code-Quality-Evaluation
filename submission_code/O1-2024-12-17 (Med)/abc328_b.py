def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    D = list(map(int, data[1:]))

    count = 0
    for i in range(1, N+1):
        for j in range(1, D[i-1]+1):
            combined = str(i) + str(j)
            if all(ch == combined[0] for ch in combined):
                count += 1

    print(count)

# Do not remove or alter the following call
if __name__ == "__main__":
    main()
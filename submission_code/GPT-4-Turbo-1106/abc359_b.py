def main():
    N = int(input().strip())
    A = list(map(int, input().strip().split()))

    count = 0
    for i in range(1, N + 1):
        first_index = A.index(i)
        second_index = A.index(i, first_index + 1)
        if second_index - first_index == 2:
            count += 1

    print(count)

if __name__ == "__main__":
    main()
def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Sort a copy to find the second largest value
    sorted_A = sorted(A)
    second_largest = sorted_A[-2]

    # Find its 1-based index in the original list
    idx = A.index(second_largest) + 1
    print(idx)

if __name__ == "__main__":
    main()
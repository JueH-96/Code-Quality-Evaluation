def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    # sort the array to get the second largest element
    sorted_A = sorted(A)
    second_largest = sorted_A[-2]
    # find the index (1-indexed)
    index = A.index(second_largest) + 1
    print(index)

if __name__ == '__main__':
    main()
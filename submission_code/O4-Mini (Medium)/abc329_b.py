def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:]))

    # Find the maximum value
    mx = max(arr)
    # Find the largest value among those not equal to mx
    second_max = max(x for x in arr if x != mx)
    print(second_max)

if __name__ == "__main__":
    main()
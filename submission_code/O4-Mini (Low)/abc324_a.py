def main():
    import sys
    data = sys.stdin.read().strip().split()
    # First value is N, the rest are the list A
    n = int(data[0])
    A = list(map(int, data[1:]))

    # If all elements are equal, the set of A will have length 1
    if len(set(A)) == 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
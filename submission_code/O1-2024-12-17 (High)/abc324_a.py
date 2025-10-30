def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    arr = list(map(int, data[1:]))

    if len(set(arr)) == 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
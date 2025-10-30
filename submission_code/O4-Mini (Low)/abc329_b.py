def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    max_val = max(a)
    # Filter out all occurrences of the maximum value
    filtered = [x for x in a if x != max_val]
    # The answer is the maximum of the remaining values
    print(max(filtered))

if __name__ == "__main__":
    main()
def main():
    import sys

    data = sys.stdin.read().strip()
    N = int(data)

    ones_count = N + 1
    zeros_count = N

    result = []
    while ones_count > 0 or zeros_count > 0:
        if ones_count > 0:
            result.append('1')
            ones_count -= 1
        if zeros_count > 0:
            result.append('0')
            zeros_count -= 1

    print("".join(result))

if __name__ == "__main__":
    main()
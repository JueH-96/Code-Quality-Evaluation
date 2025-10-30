def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    s = data[1]
    result = []
    for c in s:
        result.append(c)
        result.append(c)
    sys.stdout.write(''.join(result))

if __name__ == "__main__":
    main()
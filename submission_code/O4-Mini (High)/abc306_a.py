def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1]
    result = []
    for c in S:
        result.append(c * 2)
    print(''.join(result))

if __name__ == '__main__':
    main()
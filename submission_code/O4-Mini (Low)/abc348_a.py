def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    result = []
    for i in range(1, N+1):
        if i % 3 == 0:
            result.append('x')
        else:
            result.append('o')
    print(''.join(result))

if __name__ == "__main__":
    main()
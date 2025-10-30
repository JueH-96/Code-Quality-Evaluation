def main():
    import sys
    N = int(sys.stdin.readline().strip())
    result = []
    for i in range(1, N+1):
        if i % 3 == 0:
            result.append('x')
        else:
            result.append('o')
    print(''.join(result))

main()
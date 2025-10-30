def main():
    import sys
    N = int(sys.stdin.readline().strip())
    result = []
    for i in range(2*N + 1):
        if i % 2 == 0:
            result.append('1')
        else:
            result.append('0')
    print(''.join(result))

main()
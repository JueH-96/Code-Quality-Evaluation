def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    for x in range(N+1):
        for y in range(N+1):
            for z in range(N+1):
                if x + y + z <= N:
                    print(x, y, z)

if __name__ == '__main__':
    main()
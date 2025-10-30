def main():
    N = int(input())
    for i in range(60):
        if N & (1 << i):
            print(i)
            return

if __name__ == '__main__':
    main()
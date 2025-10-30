def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    index = 1
    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        if N % 2 == 1 or K != N // 2:
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    main()
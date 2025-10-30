import sys

def main():
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    idx = 1
    for _ in range(T):
        N = int(data[idx])
        K = int(data[idx + 1])
        idx += 2
        if N % 2 == 0:
            half = N // 2
            if K == 0 or K == half:
                print("No")
            else:
                print("Yes")
        else:
            if K == 0:
                print("No")
            else:
                print("Yes")

if __name__ == '__main__':
    main()
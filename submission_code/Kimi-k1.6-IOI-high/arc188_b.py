import sys

def main():
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    idx = 1
    for _ in range(T):
        N = int(data[idx])
        K = int(data[idx+1])
        idx += 2
        if N % 2 == 1:
            print("Yes")
        else:
            if 2 * K != N:
                print("Yes")
            else:
                print("No")

if __name__ == "__main__":
    main()
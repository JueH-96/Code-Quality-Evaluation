import sys

def main():
    input = sys.stdin.read().split()
    T = int(input[0])
    idx = 1
    for _ in range(T):
        N = int(input[idx])
        K = int(input[idx+1])
        idx +=2
        if N % 2 == 0:
            if K == N//2:
                print("No")
            else:
                print("Yes")
        else:
            print("Yes")

if __name__ == "__main__":
    main()
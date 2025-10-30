import sys

def main() -> None:
    S = sys.stdin.readline().strip()        # read the 16-character string
    for i in range(1, 16, 2):               # 0-based indices 1,3,5,... correspond to positions 2,4,6,...
        if S[i] != '0':
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()
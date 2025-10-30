import sys

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    S = [input().strip() for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            t = S[i] + S[j]
            if t == t[::-1]:
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()
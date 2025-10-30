import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    has_male = [False] * (n + 1)  # 1-based indexing
    for _ in range(m):
        a, b = sys.stdin.readline().split()
        a = int(a)
        if b == 'M' and not has_male[a]:
            print("Yes")
            has_male[a] = True
        else:
            print("No")

if __name__ == "__main__":
    main()
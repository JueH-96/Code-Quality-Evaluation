import sys

def main():
    input = sys.stdin.read
    N, L, *A = map(int, input().split())
    passed = sum(1 for score in A if score >= L)
    print(passed)

if __name__ == "__main__":
    main()
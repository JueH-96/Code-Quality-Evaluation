import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    A = list(map(int, data))
    result = 0
    for i in range(64):
        result += A[i] * (2 ** i)
    print(result)

if __name__ == "__main__":
    main()
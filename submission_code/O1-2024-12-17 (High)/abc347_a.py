def main():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    result = [a // K for a in A if a % K == 0]
    print(*result)

if __name__ == "__main__":
    main()
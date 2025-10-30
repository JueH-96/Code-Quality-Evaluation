def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    count = 0
    for _ in range(N):
        if input().strip() == "Takahashi":
            count += 1
    print(count)

if __name__ == "__main__":
    main()
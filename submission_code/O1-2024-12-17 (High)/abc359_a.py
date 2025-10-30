def main():
    N = int(input().strip())
    count_takahashi = 0
    for _ in range(N):
        s = input().strip()
        if s == "Takahashi":
            count_takahashi += 1
    print(count_takahashi)

main()
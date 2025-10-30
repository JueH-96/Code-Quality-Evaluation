def main():
    S = list(map(int, input().split()))
    if (
        all(S[i] <= S[i+1] for i in range(7)) and
        all(100 <= x <= 675 for x in S) and
        all(x % 25 == 0 for x in S)
    ):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
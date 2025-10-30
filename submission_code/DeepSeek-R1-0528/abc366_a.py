def main():
    N, T, A = map(int, input().split())
    majority = (N + 1) // 2
    if T >= majority or A >= majority:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
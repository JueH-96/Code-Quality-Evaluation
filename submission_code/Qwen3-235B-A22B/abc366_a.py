def main():
    N, T, A = map(int, input().split())
    R = N - T - A
    if T > A + R or A > T + R:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
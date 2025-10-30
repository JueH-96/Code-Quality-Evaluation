def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    dishes = [input().strip() for _ in range(N)]

    # Scan for consecutive "sweet"
    for i in range(N - 1):
        if dishes[i] == "sweet" and dishes[i + 1] == "sweet":
            # If the second sweet in the pair is not the last dish,
            # he will be unable to eat all dishes.
            if i + 1 < N - 1:
                print("No")
                return
    print("Yes")

if __name__ == "__main__":
    main()
def main():
    N = int(input().strip())
    dishes = [input().strip() for _ in range(N)]
    
    for i in range(1, N):
        if dishes[i] == "sweet" and dishes[i-1] == "sweet":
            if i < N - 1:
                print("No")
                return
    print("Yes")

if __name__ == "__main__":
    main()
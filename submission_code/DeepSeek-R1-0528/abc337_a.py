def main():
    N = int(input().strip())
    total_takahashi = 0
    total_aoki = 0
    for _ in range(N):
        X, Y = map(int, input().split())
        total_takahashi += X
        total_aoki += Y
    
    if total_takahashi > total_aoki:
        print("Takahashi")
    elif total_takahashi < total_aoki:
        print("Aoki")
    else:
        print("Draw")

if __name__ == "__main__":
    main()
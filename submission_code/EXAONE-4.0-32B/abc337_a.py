def main():
    N = int(input().strip())
    total_takahashi = 0
    total_aoki = 0
    
    for _ in range(N):
        x, y = map(int, input().split())
        total_takahashi += x
        total_aoki += y
        
    if total_takahashi > total_aoki:
        print("Takahashi")
    elif total_takahashi < total_aoki:
        print("Aoki")
    else:
        print("Draw")

if __name__ == "__main__":
    main()
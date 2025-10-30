def main():
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    total_takahashi = sum(A)
    total_aoki_before = sum(B)
    
    deficit = total_takahashi - total_aoki_before
    runs_needed = deficit + 1
    print(runs_needed)

if __name__ == "__main__":
    main()
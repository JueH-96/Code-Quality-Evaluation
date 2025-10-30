def main():
    takahashi_runs = list(map(int, input().split()))
    aoki_runs = list(map(int, input().split()))
    
    total_takahashi = sum(takahashi_runs)
    total_aoki_before_nineth = sum(aoki_runs)
    
    runs_needed = total_takahashi - total_aoki_before_nineth + 1
    print(runs_needed)

if __name__ == "__main__":
    main()
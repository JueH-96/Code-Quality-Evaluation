def main():
    M = int(input().strip())
    D_list = list(map(int, input().split()))
    
    total_days = sum(D_list)
    mid = (total_days + 1) // 2
    
    cumulative = 0
    for i in range(M):
        days_in_month = D_list[i]
        if cumulative + days_in_month >= mid:
            month = i + 1
            day = mid - cumulative
            print(f"{month} {day}")
            return
        else:
            cumulative += days_in_month

if __name__ == "__main__":
    main()
def main():
    M = int(input().strip())
    days_list = list(map(int, input().split()))
    
    total_days = sum(days_list)
    mid = (total_days + 1) // 2
    
    cumulative = 0
    for i in range(M):
        d = days_list[i]
        if cumulative + d >= mid:
            day_in_month = mid - cumulative
            print(f"{i+1} {day_in_month}")
            return
        cumulative += d

if __name__ == "__main__":
    main()
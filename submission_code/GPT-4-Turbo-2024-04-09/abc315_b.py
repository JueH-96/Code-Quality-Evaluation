import sys
input = sys.stdin.read

def main():
    data = input().split()
    M = int(data[0])
    D = list(map(int, data[1:]))
    
    total_days = sum(D)
    middle_day = (total_days + 1) // 2
    
    current_day = 0
    for month_index in range(M):
        current_day += D[month_index]
        if current_day >= middle_day:
            day_in_month = middle_day - (current_day - D[month_index])
            print(month_index + 1, day_in_month)
            break

if __name__ == "__main__":
    main()
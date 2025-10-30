import sys

def find_middle_day(months, days):
    total_days = sum(days)
    middle_day = (total_days + 1) // 2
    current_day = 0
    
    for month, day_count in enumerate(days, start=1):
        if current_day + day_count >= middle_day:
            return month, middle_day - current_day
        current_day += day_count

if __name__ == "__main__":
    input = sys.stdin.read
    data = list(map(int, input().split()))
    M = data[0]
    days = data[1:]
    month, day = find_middle_day(M, days)
    print(month, day)
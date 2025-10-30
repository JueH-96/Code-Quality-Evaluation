import sys

def find_middle_day():
    input = sys.stdin.read
    data = input().split()

    M = int(data[0])
    D = list(map(int, data[1:M+1]))

    total_days = sum(D)
    middle_day = (total_days + 1) // 2

    cumulative_days = 0
    for i in range(M):
        cumulative_days += D[i]
        if cumulative_days >= middle_day:
            month = i + 1
            day = middle_day - (cumulative_days - D[i])
            break

    print(month, day)

find_middle_day()
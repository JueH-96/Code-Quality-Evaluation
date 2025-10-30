def main():
    n = int(input().strip())
    total_days = 7 * n
    data = list(map(int, input().split()))
    
    weeks = []
    for i in range(n):
        start_index = i * 7
        end_index = start_index + 7
        week_data = data[start_index:end_index]
        total_steps = sum(week_data)
        weeks.append(str(total_steps))
    
    print(" ".join(weeks))

if __name__ == "__main__":
    main()
def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, D = map(int, data[:2])
    schedules = data[2:]
    
    # day_free[j] will be True if all people are free on day j, otherwise False
    day_free = [True]*D
    
    # Determine for each day if everyone is free
    for i in range(N):
        for j, c in enumerate(schedules[i]):
            if c == 'x':
                day_free[j] = False
    
    # Find the maximum consecutive True segments in day_free
    ans = 0
    current_streak = 0
    for j in range(D):
        if day_free[j]:
            current_streak += 1
        else:
            ans = max(ans, current_streak)
            current_streak = 0
    ans = max(ans, current_streak)
    
    print(ans)

# DO NOT forget to call main()
if __name__ == "__main__":
    main()
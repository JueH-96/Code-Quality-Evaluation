# YOUR CODE HERE
def main():
    import sys
    N, D = map(int, sys.stdin.readline().split())
    schedules = [sys.stdin.readline().strip() for _ in range(N)]
    
    # For each day, check if all are 'o'
    all_free = []
    for j in range(D):
        free = True
        for i in range(N):
            if schedules[i][j] != 'o':
                free = False
                break
        all_free.append(free)
    
    # Find the maximum consecutive True in all_free
    max_len = 0
    current = 0
    for day_free in all_free:
        if day_free:
            current +=1
            if current > max_len:
                max_len = current
        else:
            current =0
    print(max_len)

if __name__ == "__main__":
    main()
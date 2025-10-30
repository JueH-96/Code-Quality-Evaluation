# YOUR CODE HERE
def solve():
    N, A, B = map(int, input().split())
    D = list(map(int, input().split()))
    
    week_length = A + B
    
    for i in range(week_length):
        all_holidays = True
        for d in D:
            day = (d + i) % week_length
            if day >= A:
                all_holidays = False
                break
        if all_holidays:
            return "Yes"
    
    return "No"

print(solve())
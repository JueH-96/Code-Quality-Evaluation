# YOUR CODE HERE
def solve():
    M = int(input())
    D_list = list(map(int, input().split()))

    # Calculate the total number of days in the year
    S = sum(D_list)
    
    # The problem states that S (total_days_in_year) is odd.
    # The middle day is the ((S + 1) / 2)-th day of the year.
    # Since S is odd, S + 1 is even.
    # So, (S + 1) // 2 will perform an exact integer division.
    target_day_of_year = (S + 1) // 2
    
    # Iterate through the months to find which month and day
    # the target_day_of_year corresponds to.
    # The months are 1-indexed in the problem description,
    # but D_list is 0-indexed (D_list[0] for month 1, D_list[1] for month 2, etc.).
    
    # month_idx will be the 0-based index for D_list
    for month_idx in range(M):
        days_in_current_month = D_list[month_idx]
        
        if target_day_of_year <= days_in_current_month:
            # The target day falls into this month.
            # The month number (1-indexed) is month_idx + 1.
            # The day number within this month is the current value of target_day_of_year.
            result_month = month_idx + 1
            result_day = target_day_of_year
            
            # Print the result in the specified format "a b"
            print(result_month, result_day)
            
            # We found the answer, so we can stop the loop and the program.
            break
        else:
            # The target day is not in this month.
            # Subtract the number of days in the current month from target_day_of_year
            # to find its position relative to the start of the next month.
            target_day_of_year -= days_in_current_month

if __name__ == '__main__':
    solve()
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    
    # Read the a_i and b_i values
    medicines = []
    for i in range(N):
        a = int(data[2 + 2*i])
        b = int(data[3 + 2*i])
        medicines.append((a, b))
    
    # We need to find the first day where the total pills taken is <= K
    # We will use a dictionary to count the number of pills for each day
    pills_per_day = {}
    
    # Populate the dictionary with the number of pills needed for each day
    for a, b in medicines:
        if b > 0:
            if 1 in pills_per_day:
                pills_per_day[1] += b
            else:
                pills_per_day[1] = b
            
            if a+1 in pills_per_day:
                pills_per_day[a+1] -= b
            else:
                pills_per_day[a+1] = -b
    
    # Now we need to iterate over the days in sorted order and keep a running total
    current_pills = 0
    sorted_days = sorted(pills_per_day.keys())
    
    for day in sorted_days:
        current_pills += pills_per_day[day]
        
        if current_pills <= K:
            print(day)
            return

main()
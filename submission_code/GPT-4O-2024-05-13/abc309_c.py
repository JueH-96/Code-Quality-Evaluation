# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    
    medicines = []
    index = 2
    for _ in range(N):
        a = int(data[index])
        b = int(data[index + 1])
        medicines.append((a, b))
        index += 2
    
    # Create a list to store the total number of pills taken each day
    pills_per_day = {}
    
    for a, b in medicines:
        if a not in pills_per_day:
            pills_per_day[a] = 0
        pills_per_day[a] += b
    
    # Calculate the cumulative pills taken each day
    cumulative_pills = 0
    day = 1
    while True:
        if day in pills_per_day:
            cumulative_pills += pills_per_day[day]
        if cumulative_pills <= K:
            print(day)
            return
        day += 1

if __name__ == "__main__":
    main()
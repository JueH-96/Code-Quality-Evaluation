def first_day_with_k_pills_or_less(N, K, medicines):
    # Create an array to store the number of pills taken on each day
    max_days = max(a for a, b in medicines)
    pills_per_day = [0] * (max_days + 1)

    # Fill the pills_per_day array
    for a, b in medicines:
        for day in range(1, a + 1):
            pills_per_day[day] += b

    # Find the first day where the total pills are K or less
    for day in range(1, max_days + 1):
        if pills_per_day[day] <= K:
            return day

    return -1  # This should not happen according to the problem statement

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N, K = map(int, data[0].split())
    medicines = [tuple(map(int, line.split())) for line in data[1:N + 1]]
    
    result = first_day_with_k_pills_or_less(N, K, medicines)
    print(result)

if __name__ == "__main__":
    main()
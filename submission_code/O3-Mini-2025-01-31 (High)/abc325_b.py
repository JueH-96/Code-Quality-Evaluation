def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    # meeting[t] will hold the number of employees available when the meeting is held from t to t+1 UTC.
    meeting = [0] * 24
    
    index = 1
    for _ in range(n):
        w = int(data[index])
        x = int(data[index + 1])
        index += 2
        # For each possible meeting start time T (0 <= T < 24), 
        # the local time at the base is (T + x) mod 24.
        # An employee can attend if the meeting is fully within 9:00 to 18:00.
        # Since the meeting lasts one hour, the starting time must be in the range [9,17] locally.
        for T in range(24):
            local_time = (T + x) % 24
            if 9 <= local_time <= 17:
                meeting[T] += w

    # Print the maximum number of employees that can attend.
    sys.stdout.write(str(max(meeting)))
    
if __name__ == '__main__':
    main()
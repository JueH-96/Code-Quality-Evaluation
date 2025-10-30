def main():
    import sys
    import bisect
    
    data = sys.stdin.read().split()
    if not data:
        return

    N = int(data[0])
    M = int(data[1])
    fireworks_days = list(map(int, data[2:2+M]))
    
    # For each day i from 1 to N, we find the first fireworks day >= i using binary search
    output = []
    for i in range(1, N+1):
        pos = bisect.bisect_left(fireworks_days, i)
        # fireworks_days[pos] is the first fireworks day on or after day i
        wait_time = fireworks_days[pos] - i
        output.append(str(wait_time))
    
    sys.stdout.write("
".join(output))

if __name__ == '__main__':
    main()
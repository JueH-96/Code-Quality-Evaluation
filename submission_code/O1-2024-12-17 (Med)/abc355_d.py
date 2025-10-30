def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    events = []
    idx = 1
    for _ in range(N):
        l = int(input_data[idx]); r = int(input_data[idx+1])
        idx += 2
        # 0 indicates start of an interval, 1 indicates end of an interval
        events.append((l, 0))
        events.append((r, 1))
    
    # Sort by coordinate; starts (0) should come before ends (1) if same coordinate
    events.sort(key=lambda x: (x[0], x[1]))
    
    active = 0
    ans = 0
    for coordinate, etype in events:
        if etype == 0:
            # interval start
            ans += active
            active += 1
        else:
            # interval end
            active -= 1
    
    print(ans)

# Do not forget to call main()
main()
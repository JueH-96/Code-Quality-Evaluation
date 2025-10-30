import sys
import threading

def main():
    import sys
    from collections import defaultdict
    
    input = sys.stdin.readline
    N = int(input())
    
    events = []        # store (t, x) for each event
    finds = defaultdict(list)
    uses  = defaultdict(list)
    find_positions = []  # list of event-indices of find events, in order
    
    for i in range(N):
        t, x = map(int, input().split())
        events.append((t, x))
        if t == 1:
            finds[x].append(i)
            find_positions.append(i)
        else:
            uses[x].append(i)
    
    # Check feasibility: for each type x, must have at least as many finds as uses
    for x, ulist in uses.items():
        if len(ulist) > len(finds.get(x, ())):
            print(-1)
            return
    
    # Assign picks as late as possible for each type
    picked = [False] * N  # picked[i] = True if we pick at event i
    for x, ulist in uses.items():
        flist = finds[x]
        fptr = len(flist) - 1
        # iterate uses in reverse, match each to the latest available find < use_pos
        for use_pos in reversed(ulist):
            # move fptr until find[fptr] <= use_pos
            while fptr >= 0 and flist[fptr] > use_pos:
                fptr -= 1
            # now flist[fptr] is the latest eligible find
            # fptr must be >= 0 because we checked total counts earlier
            picked[flist[fptr]] = True
            fptr -= 1
    
    # Simulate to find maximum inventory
    curr = 0
    max_inv = 0
    for i, (t, x) in enumerate(events):
        if t == 1:
            if picked[i]:
                curr += 1
                if curr > max_inv:
                    max_inv = curr
        else:
            curr -= 1
            # curr should never go negative in a valid assignment
    
    # Output
    print(max_inv)
    # For each find event in order, output 1 if picked, 0 otherwise
    out = []
    for pos in find_positions:
        out.append('1' if picked[pos] else '0')
    print(" ".join(out))

if __name__ == "__main__":
    main()
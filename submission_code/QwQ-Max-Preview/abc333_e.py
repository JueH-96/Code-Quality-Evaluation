import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    events = []
    t1_events = defaultdict(list)
    t2_events = defaultdict(list)
    for i in range(N):
        t = int(input[ptr])
        x = int(input[ptr + 1])
        ptr += 2
        events.append((t, x))
        if t == 1:
            t1_events[x].append(i)
        else:
            t2_events[x].append(i)
    
    collected_events = defaultdict(set)
    for x in t2_events:
        t1 = t1_events.get(x, [])
        t2 = t2_events[x]
        if len(t1) < len(t2):
            print(-1)
            return
        t1_sorted = sorted(t1)
        t2_sorted = sorted(t2)
        j = len(t1_sorted) - 1
        collected = collected_events[x]
        for t2_pos in reversed(t2_sorted):
            while j >= 0 and t1_sorted[j] >= t2_pos:
                j -= 1
            if j < 0:
                print(-1)
                return
            collected.add(t1_sorted[j])
            j -= 1
    
    sum_collected = 0
    sum_used = 0
    peak = 0
    actions = []
    available = defaultdict(int)
    for i in range(N):
        t, x = events[i]
        if t == 1:
            if x in collected_events and i in collected_events[x]:
                sum_collected += 1
                available[x] += 1
                current = sum_collected - sum_used
                if current > peak:
                    peak = current
                actions.append(1)
            else:
                actions.append(0)
        else:
            if available[x] == 0:
                print(-1)
                return
            available[x] -= 1
            sum_used += 1
    
    output_actions = []
    for i in range(N):
        if events[i][0] == 1:
            output_actions.append(str(actions[i]))
    print(peak)
    print(' '.join(output_actions))

if __name__ == "__main__":
    main()
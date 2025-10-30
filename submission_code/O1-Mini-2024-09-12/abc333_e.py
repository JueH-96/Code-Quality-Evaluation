# YOUR CODE HERE
import sys

def main():
    import sys
    import threading

    def solve():
        import sys

        sys.setrecursionlimit(1 << 25)
        N_and_rest = sys.stdin.read().split()
        N = int(N_and_rest[0])
        events = []
        ptr = 1
        for _ in range(N):
            t = int(N_and_rest[ptr])
            x = int(N_and_rest[ptr+1])
            events.append( (t, x) )
            ptr +=2

        list_pickups = [[] for _ in range(N+1)]
        list_uses = [[] for _ in range(N+1)]

        for idx, (t, x) in enumerate(events, start=1):
            if t ==1:
                list_pickups[x].append(idx)
            elif t ==2:
                list_uses[x].append(idx)

        assigned_pickups = []
        for x in range(1, N+1):
            pickups = list_pickups[x]
            uses = list_uses[x]
            if len(pickups) < len(uses):
                print(-1)
                return
            if not uses:
                continue
            pickups_sorted = pickups
            uses_sorted = uses
            # Sort pickups and uses in ascending order
            pickups_sorted = sorted(pickups_sorted)
            uses_sorted = sorted(uses_sorted)
            idx_p = len(pickups_sorted)-1
            for use in reversed(uses_sorted):
                while idx_p >=0 and pickups_sorted[idx_p] > use:
                    idx_p -=1
                if idx_p <0:
                    print(-1)
                    return
                assigned_pickups.append( (pickups_sorted[idx_p], use) )
                idx_p -=1

        # Now, collect all interval events
        interval_events = []
        for pickup, use in assigned_pickups:
            interval_events.append( (pickup, 1) )
            interval_events.append( (use, -1) )

        # Sort interval events
        # For the same event_index, process end (-1) before start (+1)
        interval_events_sorted = sorted(interval_events, key=lambda x: (x[0], 0 if x[1]==-1 else 1))

        current =0
        K_min =0
        for event_index, delta in interval_events_sorted:
            current += delta
            if current > K_min:
                K_min = current

        # Now, mark assigned pickups
        assigned_pickup_flag = [False]*(N+1)
        for pickup, use in assigned_pickups:
            assigned_pickup_flag[pickup] = True

        # Collect choices
        choices = []
        for idx, (t, x) in enumerate(events, start=1):
            if t ==1:
                if assigned_pickup_flag[idx]:
                    choices.append('1')
                else:
                    choices.append('0')

        # Output
        print(K_min)
        print(' '.join(choices))

    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()
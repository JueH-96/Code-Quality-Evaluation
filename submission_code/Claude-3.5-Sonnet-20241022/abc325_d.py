N = int(input())
events = []

# Read input and create events list
for i in range(N):
    t, d = map(int, input().split())
    # Store (time, type, product_id)
    # type: 0 for enter, 1 for leave
    events.append((t, 0, i))
    events.append((t + d, 1, i))

# Sort events by time, with leaves coming before enters at same time
events.sort(key=lambda x: (x[0], x[1]))

available = set()  # Products currently in range
used = set()      # Products already printed
last_print = -2   # Time of last print (-2 to handle first print at time -1)
ans = 0

for time, event_type, prod_id in events:
    if event_type == 0:  # Product enters
        available.add(prod_id)
    else:  # Product leaves
        if prod_id in available:
            # If we can print (enough time passed since last print)
            if time > last_print + 1:
                # If product not already printed, print it
                if prod_id not in used:
                    used.add(prod_id)
                    last_print = time
                    ans += 1
            # If we can't print this leaving product but there are other available products
            elif available - used:
                # Find any unprinted product
                for p in available:
                    if p not in used:
                        used.add(p)
                        last_print = time - 1  # Print 1 microsecond before
                        ans += 1
                        break
        available.remove(prod_id)

print(ans)
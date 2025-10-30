N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Create events: (price, type, change)
# type 0: seller becomes available (add 1 seller)
# type 1: buyer drops out (subtract 1 buyer)
events = []

for a in A:
    events.append((a, 0, 1))  # at price a, +1 seller

for b in B:
    events.append((b + 1, 1, -1))  # at price b+1, -1 buyer (buyer drops out)

# Sort events by price, then by type (sellers first)
events.sort()

# Initial state at price 0: 0 sellers, M buyers
sellers = 0
buyers = M

# Check if condition is satisfied at price 1
if sellers >= buyers:
    print(1)
else:
    found = False
    for price, event_type, change in events:
        if event_type == 0:  # seller becomes available
            sellers += change
        else:  # buyer drops out
            buyers += change
        
        # Check condition after this event
        if sellers >= buyers:
            print(price)
            found = True
            break
    
    if not found:
        # This shouldn't happen given the problem constraints
        print(max(max(A), max(B)) + 1)
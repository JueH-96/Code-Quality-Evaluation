# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Check if all are already 1
if all(x == 1 for x in a):
    print("Yes")
else:
    # Count the number of 0s
    zero_count = a.count(0)
    
    # If there's only one 0 and it's isolated, we can't fix it
    if zero_count == 1:
        # Find the position of the single 0
        pos = a.index(0)
        # Check if it's isolated (not adjacent to any other 0)
        prev_pos = (pos - 1) % n
        next_pos = (pos + 1) % n
        if a[prev_pos] == 1 and a[next_pos] == 1:
            print("No")
        else:
            print("Yes")
    else:
        # If there are multiple 0s or no 0s, we can always make it work
        print("Yes")
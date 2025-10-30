# YOUR CODE HERE
import sys

# Read N from stdin
n = int(sys.stdin.readline())
# Read list A from stdin
a = list(map(int, sys.stdin.readline().split()))

# Initialize operation count
count = 0

# Calculate initial count of positive elements
# The process continues until A contains one or fewer positive elements.
positive_count = 0
for x in a:
    if x > 0:
        positive_count += 1

# Loop as long as there are more than 1 positive elements
while positive_count > 1:
    # Perform the operation:
    
    # 1. Sort A in descending order
    # This brings the largest elements to the front.
    a.sort(reverse=True)

    # Store the values of the two largest elements *before* decrementing.
    # These values are needed to correctly update the positive_count later.
    # Indices 0 and 1 are guaranteed to exist because the constraint N >= 2 holds.
    val0 = a[0]
    val1 = a[1] 

    # 2. Decrease the largest two elements (A_1 and A_2 after sorting) by 1.
    # The problem description implies this step is always performed if the loop condition
    # (more than one positive element) is met. It doesn't specify conditions like
    # "only if they are positive". Based on samples, we always decrease a[0] and a[1].
    a[0] -= 1
    a[1] -= 1
    
    # Increment the operation counter
    count += 1

    # Update the count of positive elements efficiently.
    # After decrementing, if an element's value was exactly 1 *before* decrementing, 
    # it has now become 0. This means the number of positive elements decreases by one.
    # We check this for both decremented elements.
    if val0 == 1:
        positive_count -= 1
    
    # The check for val1 is independent. If both val0 and val1 were 1, 
    # positive_count decreases by 2 in total (one for each).
    if val1 == 1:
        positive_count -= 1 

# Print the total number of operations performed
print(count)
# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

operations = 0

while True:
    # Count positive elements
    positive_count = sum(1 for x in a if x > 0)
    
    # If one or fewer positive elements, stop
    if positive_count <= 1:
        break
    
    # Sort in descending order
    a.sort(reverse=True)
    
    # Decrease the two largest elements by 1
    a[0] -= 1
    a[1] -= 1
    
    operations += 1

print(operations)
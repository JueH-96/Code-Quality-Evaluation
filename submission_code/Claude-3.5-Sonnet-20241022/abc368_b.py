N = int(input())
A = list(map(int, input().split()))

operations = 0
while True:
    # Count positive numbers
    positive_count = sum(1 for x in A if x > 0)
    if positive_count <= 1:
        break
        
    # Sort in descending order
    A.sort(reverse=True)
    
    # Decrease the two largest numbers by 1
    A[0] -= 1
    A[1] -= 1
    
    operations += 1

print(operations)
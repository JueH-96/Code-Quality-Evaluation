# YOUR CODE HERE
N, X, Y = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort dishes by sweetness and saltiness in descending order
sorted_A = sorted(A, reverse=True)
sorted_B = sorted(B, reverse=True)

# Find the minimum k such that we can exceed a threshold
cumsum_A = 0
cumsum_B = 0
for k in range(1, N + 1):
    cumsum_A += sorted_A[k - 1]
    cumsum_B += sorted_B[k - 1]
    
    # Check if top k by sweetness exceed X
    if cumsum_A > X:
        print(k)
        break
    # Check if top k by saltiness exceed Y
    if cumsum_B > Y:
        print(k)
        break
else:
    # If we reach here, even eating all dishes doesn't exceed either threshold
    print(N)
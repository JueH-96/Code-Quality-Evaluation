def min_dishes_to_exceed_limits(N, X, Y, A, B):
    # Sort indices based on sweetness and saltiness
    sorted_by_sweetness = sorted(range(N), key=lambda i: A[i], reverse=True)
    sorted_by_saltiness = sorted(range(N), key=lambda i: B[i], reverse=True)
    
    # Calculate minimum dishes by sweetness
    total_sweetness = 0
    min_dishes_by_sweetness = N
    for i in range(N):
        total_sweetness += A[sorted_by_sweetness[i]]
        if total_sweetness > X:
            min_dishes_by_sweetness = i + 1
            break
    
    # Calculate minimum dishes by saltiness
    total_saltiness = 0
    min_dishes_by_saltiness = N
    for i in range(N):
        total_saltiness += B[sorted_by_saltiness[i]]
        if total_saltiness > Y:
            min_dishes_by_saltiness = i + 1
            break
    
    # The answer is the minimum of the two
    return min(min_dishes_by_sweetness, min_dishes_by_saltiness)

# Reading input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])
A = list(map(int, data[3:N+3]))
B = list(map(int, data[N+3:2*N+3]))

# Get the result
result = min_dishes_to_exceed_limits(N, X, Y, A, B)

# Output the result
print(result)
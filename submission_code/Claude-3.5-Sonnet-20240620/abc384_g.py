# YOUR CODE HERE
def calculate_sum(A, B, X, Y):
    prefix_sum_A = [0]
    prefix_sum_B = [0]
    for a in A:
        prefix_sum_A.append(prefix_sum_A[-1] + a)
    for b in B:
        prefix_sum_B.append(prefix_sum_B[-1] + b)

    results = []
    for x, y in zip(X, Y):
        sum_A = prefix_sum_A[x]
        sum_B = prefix_sum_B[y]
        count_A = x
        count_B = y
        
        result = count_A * sum_B + count_B * sum_A - 2 * min(count_A, count_B) * min(sum_A, sum_B)
        results.append(result)
    
    return results

# Read input
N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
K = int(input())
X = []
Y = []
for _ in range(K):
    x, y = map(int, input().split())
    X.append(x)
    Y.append(y)

# Calculate and print results
results = calculate_sum(A, B, X, Y)
for result in results:
    print(result)
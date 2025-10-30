# YOUR CODE HERE
def solve():
    N = int(input())
    pairs = []
    for _ in range(N):
        L, R = map(int, input().split())
        pairs.append((L, R))

    min_sum = sum(L for L, R in pairs)
    max_sum = sum(R for L, R in pairs)
    
    # Check if a solution is possible
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    # Start by setting each X_i to its minimum
    X = [L for L, R in pairs]
    current_sum = min_sum
    
    # Adjust elements one by one to make the sum 0
    for i in range(N):
        if current_sum == 0:
            break
            
        L, R = pairs[i]
        max_increase = R - L
        increase = min(max_increase, -current_sum)
        X[i] += increase
        current_sum += increase
    
    print("Yes")
    print(" ".join(map(str, X)))

if __name__ == "__main__":
    solve()
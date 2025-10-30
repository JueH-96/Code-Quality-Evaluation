def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))
    
    A.sort()
    
    # Calculate the total cost if x is larger than any A_i
    total_cost_if_x_infinite = sum(A)
    
    if total_cost_if_x_infinite <= M:
        print("infinite")
        return
    
    # Binary search for the maximum x that fits the budget
    low, high = 0, A[-1]
    
    def can_fit_budget(x):
        total_cost = sum(min(x, a) for a in A)
        return total_cost <= M
    
    while low < high:
        mid = (low + high + 1) // 2
        if can_fit_budget(mid):
            low = mid
        else:
            high = mid - 1
    
    print(low)

main()
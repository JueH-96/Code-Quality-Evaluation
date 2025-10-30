# YOUR CODE HERE
def solve():
    N, D = map(int, input().split())
    
    snakes_properties = []
    for _ in range(N):
        T, L = map(int, input().split())
        snakes_properties.append((T, L))
        
    # For each day k from 1 to D
    for k_increase in range(1, D + 1):
        # Calculate the weight of each snake with its length increased by k_increase
        # and find the maximum among them.
        
        # Using a generator expression to calculate weights for the current k_increase.
        # T_snake and L_snake are the initial thickness and length of a snake.
        # L_snake + k_increase is the new length.
        # T_snake * (L_snake + k_increase) is the new weight.
        max_weight_on_day = max(T_snake * (L_snake + k_increase) for T_snake, L_snake in snakes_properties)
        
        print(max_weight_on_day)

if __name__ == '__main__':
    solve()
# YOUR CODE HERE
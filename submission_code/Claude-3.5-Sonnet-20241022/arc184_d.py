def find_removable_balls(k, balls):
    removable = set()
    x_k, y_k = balls[k]
    
    for i in balls:
        if i == k:
            continue
        x_i, y_i = balls[i]
        if (x_i < x_k and y_i < y_k) or (x_i > x_k and y_i > y_k):
            removable.add(i)
    
    return removable

def get_remaining_sets(N, balls):
    MOD = 998244353
    seen = set()
    
    def process_operations(current_balls):
        # Convert current balls to tuple for hashing
        current_tuple = tuple(sorted(current_balls))
        if current_tuple in seen:
            return
        seen.add(current_tuple)
        
        # Try each ball as the chosen one
        for k in current_balls:
            # Find balls that would be removed
            removable = find_removable_balls(k, balls)
            # Create new set of remaining balls
            new_balls = current_balls - removable
            process_operations(new_balls)
    
    # Start with all balls
    initial_balls = set(range(N))
    process_operations(initial_balls)
    
    return len(seen) % MOD

def main():
    # Read input
    N = int(input())
    balls = {}
    
    # Read coordinates
    for i in range(N):
        x, y = map(int, input().split())
        balls[i] = (x, y)
    
    # Calculate and print result
    result = get_remaining_sets(N, balls)
    print(result)

if __name__ == "__main__":
    main()
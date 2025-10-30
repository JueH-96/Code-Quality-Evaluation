# YOUR CODE HERE
def max_tower_height():
    N = int(input())
    giants = []
    for _ in range(N):
        A, B = map(int, input().split())
        giants.append((A, B))
    
    # Sort giants by (B-A) in descending order
    giants.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    total_height = 0
    for A, B in giants:
        total_height += A
    
    # Add the head height of the topmost giant
    total_height += giants[-1][1] - giants[-1][0]
    
    return total_height

print(max_tower_height())
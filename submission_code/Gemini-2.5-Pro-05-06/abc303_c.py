import sys

def main():
    N, M, H, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    item_locations = set()
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        item_locations.add((x, y))
        
    current_x = 0
    current_y = 0
    current_health = H
    
    can_complete_all_moves = True 
    
    for i in range(N):
        move_char = S[i]
        
        # Step 1: Update coordinates based on move_char
        # (As per problem: "he ... move[s] to the following point")
        if move_char == 'R':
            current_x += 1
        elif move_char == 'L':
            current_x -= 1
        elif move_char == 'U':
            current_y += 1
        elif move_char == 'D': 
            current_y -= 1
        
        # Step 2: Consume 1 health for the move
        # (As per problem: "He consumes a health of 1 to move")
        current_health -= 1
        
        # Step 3: Check for collapse
        # (As per problem: "If Takahashi's health has become negative, he collapses")
        if current_health < 0:
            can_complete_all_moves = False
            break # Takahashi collapses, stop moving
            
        # Step 4: (Only if not collapsed, i.e., health >= 0)
        # Check for item at the new location.
        # (As per problem: "Otherwise, if an item is placed at the point he has moved to, 
        # and his health is strictly less than K, then he consumes the item")
        if (current_x, current_y) in item_locations:
            if current_health < K:
                current_health = K
                # Item is consumed and removed
                item_locations.remove((current_x, current_y))
                
    if can_complete_all_moves:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()
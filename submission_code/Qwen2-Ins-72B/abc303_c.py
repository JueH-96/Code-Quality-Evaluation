from collections import defaultdict

def main():
    N, M, H, K = map(int, input().split())
    S = input()
    items = [tuple(map(int, input().split())) for _ in range(M)]
    
    # Convert items to a dictionary for O(1) lookups
    item_dict = defaultdict(int)
    for x, y in items:
        item_dict[(x, y)] = K
    
    x, y = 0, 0
    for i, move in enumerate(S):
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        elif move == 'D':
            y -= 1
        
        H -= 1
        if H < 0:
            print("No")
            return
        
        if (x, y) in item_dict and H < K:
            H = K
            del item_dict[(x, y)]  # Remove the item after use
    
    print("Yes")

main()
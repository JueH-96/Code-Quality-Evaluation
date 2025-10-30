# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    T = int(data[0])
    index = 1
    results = []
    
    for _ in range(T):
        K = int(data[index])
        S_x = int(data[index + 1])
        S_y = int(data[index + 2])
        T_x = int(data[index + 3])
        T_y = int(data[index + 4])
        index += 5
        
        # Calculate the tile coordinates for the start point
        S_i = S_x // K
        S_j = S_y // K
        if (S_i % 2 == S_j % 2):
            S_k = S_y % K
        else:
            S_k = S_x % K
        
        # Calculate the tile coordinates for the target point
        T_i = T_x // K
        T_j = T_y // K
        if (T_i % 2 == T_j % 2):
            T_k = T_y % K
        else:
            T_k = T_x % K
        
        # Calculate the Manhattan distance in the tile space
        distance = abs(S_i - T_i) + abs(S_j - T_j) + abs(S_k - T_k)
        results.append(distance)
    
    for result in results:
        print(result)
def solve(N, X, K):
    # Calculate the level (depth) of X
    levelX = X.bit_length() - 1
    
    result = 0
    
    # Category 1: K-th ancestor of X (if it exists)
    if K <= levelX:
        ancestor = X >> K
        if ancestor >= 1:
            result += 1
    
    # Category 2: Descendants of X at a distance K
    start = X << K
    if start <= N:
        count = min(1 << K, N - start + 1)
        result += count
    
    # Category 3: Other vertices at a distance K
    for u in range(1, min(K, levelX) + 1):
        ancestor = X >> u
        if ancestor < 1:
            continue
        
        child_on_path = X >> (u - 1)
        
        if child_on_path & 1 == 0:  # X is descendant of left child of ancestor
            other_child = 2 * ancestor + 1  # Right child
            if u == K - 1:  # The other child itself
                if other_child <= N:
                    result += 1
            elif u < K - 1:  # Descendants of the other child
                depth_below = K - u - 1
                start = other_child << depth_below
                if start <= N:
                    count = min(1 << depth_below, N - start + 1)
                    result += count
        else:  # X is descendant of right child of ancestor
            other_child = 2 * ancestor  # Left child
            if u == K - 1:  # The other child itself
                if other_child <= N:
                    result += 1
            elif u < K - 1:  # Descendants of the other child
                depth_below = K - u - 1
                start = other_child << depth_below
                if start <= N:
                    count = min(1 << depth_below, N - start + 1)
                    result += count
    
    return result

def main():
    T = int(input())
    for _ in range(T):
        N, X, K = map(int, input().split())
        print(solve(N, X, K))

if __name__ == "__main__":
    main()
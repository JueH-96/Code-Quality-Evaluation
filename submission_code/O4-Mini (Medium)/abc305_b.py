def main():
    # Distances between consecutive points A-B, B-C, C-D, D-E, E-F, F-G
    distances = [3, 1, 4, 1, 5, 9]
    
    # Read the two points
    p, q = input().split()
    
    # Map letters A-G to indices 0-6
    idx_p = ord(p) - ord('A')
    idx_q = ord(q) - ord('A')
    
    # Ensure idx_p <= idx_q for summation; swap if necessary
    if idx_p > idx_q:
        idx_p, idx_q = idx_q, idx_p
    
    # Sum distances between idx_p and idx_q
    # distances[i] is the distance between point i and i+1
    result = sum(distances[idx_p:idx_q])
    
    # Output the result
    print(result)

# Call main to execute
main()
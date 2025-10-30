# YOUR CODE HERE
def find_kth_closest_distances(N, Q, a_coords, queries):
    results = []
    for b, k in queries:
        # Calculate all distances from b to each a_i
        distances = [abs(a - b) for a in a_coords]
        # Sort the distances
        distances.sort()
        # Get the k-th closest distance (1-based index)
        results.append(distances[k - 1])
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    a_coords = list(map(int, data[2:N+2]))
    
    queries = []
    index = N+2
    for _ in range(Q):
        b = int(data[index])
        k = int(data[index+1])
        queries.append((b, k))
        index += 2
    
    results = find_kth_closest_distances(N, Q, a_coords, queries)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
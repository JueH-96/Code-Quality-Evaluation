import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    edges = []
    index = 1
    for _ in range(N-1):
        A = int(data[index])
        B = int(data[index+1])
        C = int(data[index+2])
        edges.append((A, B, C))
        index += 3
    
    # To find the minimum travel distance to visit all cities, we need to sum up all the road lengths
    # because the graph is a tree (N nodes and N-1 edges), and to visit all nodes at least once in a tree,
    # the minimum distance is simply the sum of all edge weights.
    total_distance = sum(c for _, _, c in edges)
    
    # Output the result
    print(total_distance)

if __name__ == "__main__":
    main()
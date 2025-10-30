import sys

# It's good practice to increase the recursion limit for deep recursion,
# which can happen in DSU without path compression on skewed inputs.
# With path compression, the recursion depth is very low (inverse Ackermann),
# but setting it high is a safe practice in competitive programming.
sys.setrecursionlimit(4 * 10**5 + 50)

class DSU:
    """
    Disjoint Set Union data structure with an extension to track the top K
    largest elements in each set.
    """
    def __init__(self, n: int, k_max: int = 10):
        """
        Initializes the DSU for n elements (1-based indexing).
        Each element is initially in its own set.
        """
        self.n = n
        self.k_max = k_max
        # parent[i] stores the parent of element i.
        self.parent = list(range(n + 1))
        # size[i] stores the number of elements in the set rooted at i.
        self.size = [1] * (n + 1)
        # top_k[i] stores a sorted list of up to k_max largest elements
        # in the set rooted at i.
        self.top_k = [[i] for i in range(n + 1)]

    def find(self, i: int) -> int:
        """
        Finds the representative (root) of the set containing element i
        with path compression.
        """
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int):
        """
        Merges the sets containing elements i and j using union-by-size.
        Also merges their top_k lists.
        """
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by size: merge the smaller set into the larger one.
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i

            # Merge component of root_j into component of root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]

            # Merge the top_k lists
            list_i = self.top_k[root_i]
            list_j = self.top_k[root_j]
            
            merged_list = []
            p1, p2 = 0, 0

            # Standard merge of two sorted lists, taking the top k_max elements.
            while len(merged_list) < self.k_max:
                if p1 < len(list_i) and (p2 >= len(list_j) or list_i[p1] > list_j[p2]):
                    merged_list.append(list_i[p1])
                    p1 += 1
                elif p2 < len(list_j):
                    merged_list.append(list_j[p2])
                    p2 += 1
                else:  # Both lists are exhausted
                    break

            self.top_k[root_i] = merged_list
            # Free up memory for the merged-away component
            self.top_k[root_j] = None

    def query_kth_largest(self, v: int, k: int) -> int:
        """
        Finds the k-th largest element in the component containing v.
        """
        root_v = self.find(v)
        top_list = self.top_k[root_v]

        if k > len(top_list):
            return -1
        else:
            return top_list[k - 1]

def solve():
    """
    Main function to read input, process queries, and print output.
    """
    # Use fast I/O
    input = sys.stdin.readline

    try:
        N, Q = map(int, input().split())
    except (IOError, ValueError):
        # Handle empty input case, e.g., when run without redirection.
        return

    # k_max is fixed at 10 based on problem constraints
    dsu = DSU(N, k_max=10)

    # Use a buffer for output to speed up printing
    output_buffer = []

    for _ in range(Q):
        line = input().split()
        query_type = int(line[0])

        if query_type == 1:
            u, v = int(line[1]), int(line[2])
            dsu.union(u, v)
        else:  # query_type == 2
            v, k = int(line[1]), int(line[2])
            result = dsu.query_kth_largest(v, k)
            output_buffer.append(str(result))
    
    # Print all results at once for efficiency
    print("
".join(output_buffer))

if __name__ == "__main__":
    solve()
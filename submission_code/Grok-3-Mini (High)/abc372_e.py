import sys
sys.setrecursionlimit(1000000)

data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

# Initialize Union-Find
parent = [0] * (N + 1)
for i in range(N + 1):
    parent[i] = i
size_arr = [1] * (N + 1)
component_sets = {i: [i] for i in range(1, N + 1)}

# Define find function with path compression
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

# Define merge_desc function to merge two descending sorted lists
def merge_desc(a, b):
    res = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] >= b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    if i < len(a):
        res.extend(a[i:])
    else:
        res.extend(b[j:])
    return res

# Process Q queries
for _ in range(Q):
    query_type = int(data[index])
    index += 1
    if query_type == 1:
        u = int(data[index])
        index += 1
        v = int(data[index])
        index += 1
        rootU = find(u)
        rootV = find(v)
        if rootU != rootV:
            if size_arr[rootU] < size_arr[rootV]:
                parent[rootU] = rootV
                size_arr[rootV] += size_arr[rootU]
                listV = component_sets[rootV]
                listU = component_sets[rootU]
                new_list = merge_desc(listV, listU)
                component_sets[rootV] = new_list
                del component_sets[rootU]
            else:
                parent[rootV] = rootU
                size_arr[rootU] += size_arr[rootV]
                listU = component_sets[rootU]
                listV = component_sets[rootV]
                new_list = merge_desc(listU, listV)
                component_sets[rootU] = new_list
                del component_sets[rootV]
    elif query_type == 2:
        v_query = int(data[index])
        index += 1
        k = int(data[index])
        index += 1
        root = find(v_query)
        if size_arr[root] < k:
            print(-1)
        else:
            print(component_sets[root][k-1])
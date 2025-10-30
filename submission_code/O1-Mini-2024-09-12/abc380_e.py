# YOUR CODE HERE
def solve():
    import sys
    sys.setrecursionlimit(1 << 25)
    from sys import stdin
    data = stdin.read().split()
    index = 0
    N = int(data[index]); index +=1
    Q = int(data[index]); index +=1

    parent = list(range(N+2))  # parent[1..N]
    size = [1]*(N+2)
    color = list(range(N+2))  # color[1..N]
    counts = [0]*(N+2)
    for i in range(1, N+1):
        counts[i] =1

    def find(x):
        while parent[x] !=x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    output = []
    for _ in range(Q):
        query_type = data[index]; index +=1
        if query_type == '1':
            x = int(data[index]); index +=1
            c = int(data[index]); index +=1
            set_x = find(x)
            if color[set_x] != c:
                counts[color[set_x]] -= size[set_x]
                counts[c] += size[set_x]
                color[set_x] = c
                # Check left neighbor
                if x >1:
                    set_left = find(x-1)
                    if color[set_left] == c and set_left != set_x:
                        parent[set_left] = set_x
                        size[set_x] += size[set_left]
                # Check right neighbor
                if x <N:
                    set_right = find(x+1)
                    if color[set_right] == c and set_right != set_x:
                        parent[set_right] = set_x
                        size[set_x] += size[set_right]
        elif query_type == '2':
            c = int(data[index]); index +=1
            if 0 <= c <= N:
                output.append(str(counts[c]))
            else:
                output.append('0')
    print('
'.join(output))
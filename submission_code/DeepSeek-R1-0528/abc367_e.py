import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    X = list(map(int, data[2:2+n]))
    A = list(map(int, data[2+n:2+2*n]))
    
    for i in range(n):
        X[i] -= 1
        
    LOG = 60
    next_node = [[0] * n for _ in range(LOG)]
    
    next_node[0] = X
    
    for j in range(1, LOG):
        for i in range(n):
            next_node[j][i] = next_node[j-1][next_node[j-1][i]]
            
    res_index = [0] * n
    for i in range(n):
        node = i
        for bit in range(LOG-1, -1, -1):
            if (k >> bit) & 1:
                node = next_node[bit][node]
        res_index[i] = node
        
    result_arr = [A[node] for node in res_index]
    print(" ".join(map(str, result_arr)))

if __name__ == '__main__':
    main()
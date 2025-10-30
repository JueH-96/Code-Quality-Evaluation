import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    B = list(map(int, data[2+n:2+n+m]))
    
    MAX_VAL = 200000
    INF = 10**9
    
    min_index_arr = [INF] * (MAX_VAL + 1)
    
    for i, a_val in enumerate(A):
        if i < min_index_arr[a_val]:
            min_index_arr[a_val] = i
            
    F_arr = [INF] * (MAX_VAL + 1)
    F_arr[0] = min_index_arr[0]
    for v in range(1, MAX_VAL + 1):
        F_arr[v] = min(F_arr[v - 1], min_index_arr[v])
    
    output_lines = []
    for b in B:
        if F_arr[b] == INF:
            output_lines.append("-1")
        else:
            output_lines.append(str(F_arr[b] + 1))
            
    print("
".join(output_lines))

if __name__ == "__main__":
    main()
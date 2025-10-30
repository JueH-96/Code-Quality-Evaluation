import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    Q = int(data[idx])
    idx += 1
    
    plants = []
    start = 0
    global_time = 0
    output = []
    
    for _ in range(Q):
        query = data[idx]
        idx += 1
        if query == '1':
            plants.append(global_time)
        elif query == '2':
            T = int(data[idx])
            idx += 1
            global_time += T
        else:
            H = int(data[idx])
            idx += 1
            X = global_time - H
            idx_bisect = bisect.bisect_right(plants, X)
            count = idx_bisect - start
            output.append(str(count))
            start = idx_bisect
    
    print('
'.join(output))

if __name__ == "__main__":
    main()
import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    Q = int(data[idx])
    idx += 1
    
    total_t = 0
    S_list = []
    
    output = []
    
    for _ in range(Q):
        query = data[idx]
        if query == '1':
            S_list.append(total_t)
            idx += 1
        elif query == '2':
            T = int(data[idx+1])
            total_t += T
            idx += 2
        elif query == '3':
            H = int(data[idx+1])
            threshold = total_t - H
            count = bisect.bisect_right(S_list, threshold)
            output.append(str(count))
            # Remove the first 'count' elements
            S_list = S_list[count:]
            idx += 2
    
    print('
'.join(output))

if __name__ == '__main__':
    main()
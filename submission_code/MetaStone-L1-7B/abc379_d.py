import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    Q = int(data[idx])
    idx += 1
    
    type_1_queries = []
    type_2_sum = 0
    prefix_sums = []
    
    output = []
    
    for _ in range(Q):
        query = data[idx]
        if query == '1':
            t = len(data) // 2 + 1  # Assuming queries are in order, but this may not be correct
            # Compute current query number k
            k = len(data) // 2 + 1
            current_prefix_sum = type_2_sum
            type_1_queries.append((k, current_prefix_sum))
            # Insert into prefix_sums
            bisect.insort(prefix_sums, current_prefix_sum)
            idx += 1
        elif query == '2':
            T = int(data[idx])
            type_2_sum += T
            idx += 1
        elif query == '3':
            H = int(data[idx])
            idx += 1
            current_prefix_sum = type_2_sum
            threshold = current_prefix_sum - H
            if threshold < 0:
                output.append('0')
                continue
            # Binary search for threshold in prefix_sums
            count = bisect.bisect_right(prefix_sums, threshold)
            output.append(str(count))
    
    print('
'.join(output))

if __name__ == '__main__':
    main()
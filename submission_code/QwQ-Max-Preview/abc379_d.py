import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    Q = int(data[idx])
    idx += 1
    
    current_growth = 0
    bases = []
    ptr = 0
    
    for _ in range(Q):
        query_type = data[idx]
        idx += 1
        if query_type == '1':
            bases.append(current_growth)
        elif query_type == '2':
            T = int(data[idx])
            idx += 1
            current_growth += T
        else:
            H = int(data[idx])
            idx += 1
            threshold = current_growth - H
            i = bisect.bisect_right(bases, threshold, ptr, len(bases))
            count = i - ptr
            ptr = i
            print(count)

if __name__ == "__main__":
    main()
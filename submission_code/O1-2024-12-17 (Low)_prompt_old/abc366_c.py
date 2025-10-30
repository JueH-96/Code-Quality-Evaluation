def solve():
    import sys

    input_data = sys.stdin.read().strip().split()
    Q = int(input_data[0])
    
    from collections import defaultdict
    freq = defaultdict(int)
    distinct_count = 0
    
    idx = 1
    out = []
    for _ in range(Q):
        query_type = int(input_data[idx])
        idx += 1
        
        if query_type == 1:
            x = int(input_data[idx])
            idx += 1
            if freq[x] == 0:
                distinct_count += 1
            freq[x] += 1
            
        elif query_type == 2:
            x = int(input_data[idx])
            idx += 1
            freq[x] -= 1
            if freq[x] == 0:
                distinct_count -= 1
                
        else:
            # query_type == 3
            out.append(str(distinct_count))
    
    print("
".join(out))

def main():
    solve()

if __name__ == "__main__":
    main()
def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    Q = int(input_data[0])
    idx = 1

    from collections import defaultdict
    counts = defaultdict(int)
    distinct_count = 0

    answers = []
    for _ in range(Q):
        query_type = input_data[idx]
        idx += 1
        
        if query_type == '1':
            x = int(input_data[idx])
            idx += 1
            if counts[x] == 0:
                distinct_count += 1
            counts[x] += 1
        
        elif query_type == '2':
            x = int(input_data[idx])
            idx += 1
            counts[x] -= 1
            if counts[x] == 0:
                distinct_count -= 1
        
        else:  # query_type == '3'
            answers.append(str(distinct_count))
    
    print("
".join(answers))

# Do not forget to call main() !
if __name__ == "__main__":
    main()
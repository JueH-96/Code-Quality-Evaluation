def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    Q = int(input_data[0])
    index = 1
    
    from collections import defaultdict
    freq = defaultdict(int)
    distinct_count = 0
    
    for _ in range(Q):
        t = int(input_data[index])
        if t == 1:
            x = int(input_data[index+1])
            index += 2
            if freq[x] == 0:
                distinct_count += 1
            freq[x] += 1
        elif t == 2:
            x = int(input_data[index+1])
            index += 2
            freq[x] -= 1
            if freq[x] == 0:
                distinct_count -= 1
        else:  # t == 3
            index += 1
            print(distinct_count)

# Do NOT forget to call the main() function
if __name__ == "__main__":
    main()
def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    Q = int(input_data[0])
    
    A = []
    index = 1
    count_type2 = 0
    
    for _ in range(Q):
        query_type = int(input_data[index])
        index += 1
        
        if query_type == 1:
            x = int(input_data[index])
            index += 1
            A.append(x)
        else:  # query_type == 2
            k = int(input_data[index])
            index += 1
            # k-th from the end is A[-k]
            print(A[-k])
            count_type2 += 1

# Do not forget to call main()
main()
def main():
    import sys
    input = sys.stdin.readline
    
    Q = int(input())
    # Use a dictionary to maintain counts of each integer in the bag.
    bag = {}
    
    output_lines = []
    for _ in range(Q):
        query = input().split()
        if query[0] == '1':
            x = int(query[1])
            if x in bag:
                bag[x] += 1
            else:
                bag[x] = 1
        elif query[0] == '2':
            x = int(query[1])
            bag[x] -= 1
            if bag[x] == 0:
                del bag[x]
        else:  # query type 3
            output_lines.append(str(len(bag)))
    
    sys.stdout.write("
".join(output_lines))
    
if __name__ == '__main__':
    main()